import requests
import base64
import time

URL = "http://127.0.0.1:8000/api/v1/liveness"

def test_image(image_path, expected_real):
    print(f"\n--- Testing {image_path} ---")
    try:
        with open(image_path, 'rb') as f:
            base64_data = base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to read image: {e}")
        return False
        
    payload = {"image_base64": base64_data}
    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Failed to call API: {e}")
        return False
        
    print(f"Response: {data}")
    
    if data['code'] != 200:
        print(f"API Error: {data['msg']}")
        return False
        
    faces = data['data']['faces']
    if len(faces) == 0:
        print("No faces detected!")
        return False
        
    for i, face in enumerate(faces):
        is_real = face['is_real']
        score = face['score']
        print(f"Face {i+1}: is_real={is_real}, score={score:.4f}, box={face['box']}")
        if is_real != expected_real:
            print(f"TEST FAILED. Expected is_real={expected_real}, but got {is_real}")
            return False
            
    print("TEST PASSED.")
    return True

if __name__ == "__main__":
    # Wait for server to be up
    time.sleep(2)
    
    real_img = "./images/sample/image_T1.jpg"
    fake_img = "./images/sample/image_F1.jpg"
    
    all_passed = True
    all_passed &= test_image(real_img, expected_real=True)
    all_passed &= test_image(fake_img, expected_real=False)
    
    if all_passed:
        print("\nALL TESTS PASSED! Ready for Git commit.")
    else:
        print("\nSOME TESTS FAILED.")
