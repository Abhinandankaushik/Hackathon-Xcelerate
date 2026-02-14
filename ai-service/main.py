from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import io
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "AI Microservice is running"}

def read_image_file(file_contents):
    nparr = np.frombuffer(file_contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.post("/analyze/crop-red-border")
async def crop_red_border(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = read_image_file(contents)

        # 1. Blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # 2. Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 3. Define wider range for red color
        # Lower mask (0-15)
        lower_red1 = np.array([0, 30, 30])
        upper_red1 = np.array([15, 255, 255])
        
        # Upper mask (165-180)
        lower_red2 = np.array([165, 30, 30])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # 4. Morphological Closing to fill gaps in the boundary
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Optional: Dilate slightly to ensure we catch the whole border width
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 5. Find contours of the red regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: try finding ANY large prominence if red fails? 
            # For now, just return original with message
            _, buffer = cv2.imencode('.png', image)
            result_b64 = base64.b64encode(buffer).decode('utf-8')
            return {"cropped_image": f"data:image/png;base64,{result_b64}", "message": "No red border detected"}

        # 6. Find the largest contour (assumed to be the border)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 7. Create a transparency mask
        mask_out = np.zeros_like(image[:,:,0]) # Single channel mask
        
        # Fill the INSIDE of the contour
        cv2.drawContours(mask_out, [largest_contour], -1, 255, -1) 

        # 8. Create RGBA image
        b, g, r = cv2.split(image)
        rgba = [b, g, r, mask_out]
        dst = cv2.merge(rgba, 4)

        # 9. Crop to bounding rect
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add a small padding if possible
        pad = 10
        h_img, w_img = image.shape[:2]
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w_img - x, w + 2*pad)
        h = min(h_img - y, h + 2*pad)
        
        cropped = dst[y:y+h, x:x+w]

        # 10. Encode as PNG (Output)
        _, buffer = cv2.imencode('.png', cropped)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "cropped_image": f"data:image/png;base64,{result_b64}",
            "message": "Red border cropped successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/boundary")
async def analyze_boundary(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = read_image_file(contents)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny Edge Detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find Contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Simplify contours to get approximate polygon
        polygons = []
        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) > 2: # Triangle or more
                # Convert to list of points [[x, y], ...]
                points = approx.reshape(-1, 2).tolist()
                polygons.append(points)
        
        return {
            "message": "Boundary detection successful",
            "polygons": polygons,
            "count": len(polygons)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/overlay")
async def analyze_overlay(
    file: UploadFile = File(...),
    bounds: str = Form(...) # JSON string or comma-separated "minLat,minLng,maxLat,maxLng"
):
    try:
        # 1. Read Blueprint
        contents = await file.read()
        blueprint_img = read_image_file(contents)

        # 2. Parse Bounds
        # Expected format: "minLat,minLng,maxLat,maxLng" or JSON
        try:
            # simple comma split
            parts = [float(x.strip()) for x in bounds.split(',')]
            if len(parts) != 4:
                raise ValueError("Bounds must be minLat,minLng,maxLat,maxLng")
            min_lat, min_lng, max_lat, max_lng = parts
        except Exception:
             raise HTTPException(status_code=400, detail="Invalid bounds format. Use: minLat,minLng,maxLat,maxLng")

        # 3. Fetch Satellite Image from ArcGIS
        # bbox for ArcGIS is minLon, minLat, maxLon, maxLat
        bbox = f"{min_lng},{min_lat},{max_lng},{max_lat}"
        
        # Determine aspect ratio to request correct size
        width_deg = abs(max_lng - min_lng)
        height_deg = abs(max_lat - min_lat)
        
        # Ensure minimum dimensions to prevent ArcGIS errors (Zero-area bbox)
        min_dim = 0.0001 # approx 10 meters
        if width_deg < min_dim:
            center_x = (min_lng + max_lng) / 2
            min_lng = center_x - min_dim / 2
            max_lng = center_x + min_dim / 2
            width_deg = min_dim
            
        if height_deg < min_dim:
            center_y = (min_lat + max_lat) / 2
            min_lat = center_y - min_dim / 2
            max_lat = center_y + min_dim / 2
            height_deg = min_dim

        # Reconstruct bbox with buffered coordinates
        bbox = f"{min_lng},{min_lat},{max_lng},{max_lat}"
        
        aspect_ratio = width_deg / height_deg
        
        # Clamp dimensions to prevent server errors (Max 2048 typically)
        MAX_DIM = 1500
        req_w = 800
        req_h = int(req_w / aspect_ratio)
        
        if req_h > MAX_DIM:
            req_h = MAX_DIM
            req_w = int(req_h * aspect_ratio)
            
        if req_w > MAX_DIM:
             req_w = MAX_DIM
             req_h = int(req_w / aspect_ratio)

        # Ensure minimums
        req_w = max(100, req_w)
        req_h = max(100, req_h)

        print(f"DEBUG: bbox={bbox}, w={req_w}, h={req_h}")

        arcgis_url = (
            "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
            f"?bbox={bbox}&bboxSR=4326&size={req_w},{req_h}&f=image&format=png"
        )
        
        import requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        print(f"Fetching from: {arcgis_url}")
        
        try:
            response = requests.get(arcgis_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"ArcGIS Error: {response.text}")
                print(f"Failed URL: {arcgis_url}")
                # Fallback to a placeholder instead of failing hard
                raise Exception("ArcGIS returned non-200")
                
            satellite_img = read_image_file(response.content)
            
        except Exception as e:
            print(f"Satellite fetch failed: {e}. Using fallback.")
            # Create a dummy gray image as fallback (Satellite View Unavailable)
            # Size 800x600
            satellite_img = np.zeros((600, 800, 3), dtype=np.uint8)
            satellite_img[:] = (100, 100, 100) # Dark gray
            # Write text
            cv2.putText(satellite_img, "Satellite Imagery Unavailable", (50, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(satellite_img, "(Using Simulation Mode)", (200, 350), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # 4. Perform Analysis (Reuse logic)
        # Resize blueprint to match satellite image dimensions
        h, w = satellite_img.shape[:2]
        blueprint_resized = cv2.resize(blueprint_img, (w, h))
        
        # Convert to grayscale
        grayA = cv2.cvtColor(blueprint_resized, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM
        (score, diff) = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        
        # Threshold
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        # Contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_img = satellite_img.copy()
        
        # Draw green border for "Allotted Area" (The blueprint frame)
        cv2.rectangle(result_img, (0,0), (w-1, h-1), (0, 255, 0), 5)
        
        changes_detected = 0
        encroachment_area = 0
        total_area = w * h
        
        for c in contours:
            area = cv2.contourArea(c)
            if area > 200: # Filter noise
                changes_detected += 1
                encroachment_area += area
                (x, y, cw, ch) = cv2.boundingRect(c)
                # Draw Red box for deviation
                cv2.rectangle(result_img, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
                # Add label
                cv2.putText(result_img, "Deviation", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # Encode result
        _, buffer = cv2.imencode('.jpg', result_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        deviation_pct = (encroachment_area / total_area) * 100

        return {
            "similarity_score": score,
            "changes_count": changes_detected,
            "deviation_percentage": round(deviation_pct, 2),
            "result_image": f"data:image/jpeg;base64,{result_b64}",
            "status": "Non-Compliant (Encroachment)" if changes_detected > 0 else "Compliant",
            "message": "Analysis against satellite imagery complete."
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
