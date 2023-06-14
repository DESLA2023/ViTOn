#cv2.imwrite("./output/final_img.jpg", img)

# Setting up Flask
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os, urllib.request
import subprocess

# Create Flask server
app = Flask(__name__)

# Setting up directories for uploading images and storing results
UPLOAD_FOLDER = '/home/louischoi/tryHRViton/TryYours-Virtual-Try-On/static'
RESULT_FOLDER = '/home/louischoi/tryHRViton/TryYours-Virtual-Try-On/output' # /final_img.jpg
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

input_name_customer_image = "origin_web.jpg"
input_name_product_image = "cloth_web.jpg"

@app.route('/fit', methods=['POST'])
def process_image():
    
    # Delete previous input images
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
    
    # Extract images from received formData and save securely
    body_image = request.files['customer_image']
    # body_filename = secure_filename(body_image.filename)
    # body_image.save(os.path.join(app.config['UPLOAD_FOLDER'], body_filename))
    
    # input_body_path = os.path.join(app.config['UPLOAD_FOLDER'], input_name_customer_image)####################
    # print("#################### flag 1 ######################")
    body_image.save(os.path.join(app.config['UPLOAD_FOLDER'], input_name_customer_image))
    
    # print("#################### flag 2 ######################")

    product_url = request.form['product']
    print(product_url) ############### test
    
    product_save_path = os.path.join(app.config['UPLOAD_FOLDER'], input_name_product_image)
    
    # 이미지 요청 및 다운로드
    urllib.request.urlretrieve(product_url, product_save_path)
    
    
    # product_image = request.files['product']
    # product_filename = secure_filename(product_image.filename)
    # product_image.save(os.path.join(app.config['UPLOAD_FOLDER'], product_filename))

    # Running the HR-VITON fitting algorithm and getting result filename
    # result_filename = run_hr_viton(body_filename, product_filename)
    result_filename = "final_img.jpg"
    
    result_img_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    
    # Delete previous output images
    for file in os.listdir(app.config['RESULT_FOLDER']):
        os.remove(os.path.join(app.config['RESULT_FOLDER'], file))
    
    print("=== Operating HR-ViTOn Model ... ===\n")
    # terminnal_command = f"CUDA_VISIBLE_DEVICES=0 python /home/louischoi/tryHRViton/TryYours-Virtual-Try-On/main.py --result_image_path {result_img_path}" 
    terminnal_command = f"CUDA_VISIBLE_DEVICES=0 /home/louischoi/miniconda3/envs/tryHR/bin/python /home/louischoi/tryHRViton/TryYours-Virtual-Try-On/main.py" 
    os.system(terminnal_command)
    
    
    # result_img_path =  run_hr_viton(result_img_path)

    # Send result filename as response
    return {'result_path': result_img_path}

# def run_hr_viton(body_filename, product_filename):
#     # Call the Python script here. This part may need to be adjusted according to how your script is set up to receive parameters
#     result_filename = "result.jpg"
#     subprocess.run(["python", "main.py",
#                     os.path.join(app.config['UPLOAD_FOLDER'], body_filename),
#                     os.path.join(app.config['UPLOAD_FOLDER'], product_filename),
#                     os.path.join(app.config['RESULT_FOLDER'], result_filename)])
#     return result_filename


# def run_hr_viton(result_img_path):
#     # Call the Python script here. This part may need to be adjusted according to how your script is set up to receive parameters
#     # result_filename = "final_img.jpg"
    
#     # result_img_path = os.path.join(app.config['RESULT_FOLDER'], result_file_name)
    
#     subprocess.run(["CUDA_VISIBLE_DEVICES=0 ", "python ", "/home/louischoi/tryHRViton/TryYours-Virtual-Try-On/main.py ", result_img_path])
    
#     return result_img_path



@app.route('/get_result/<filename>', methods=['GET'])
def get_result(filename):
    # Send the result file with the given filename
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), mimetype='image/jpg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=18000)


