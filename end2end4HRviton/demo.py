# import os
# import shutil
# from google.colab import files

# Uncomment the following block if you would like to upload your own cloth images. 

# input_dir = 'static'
# uploaded = files.upload()
# for filename in uploaded.keys():
#   input_path = os.path.join(input_dir, filename)
#   shutil.move(filename, input_path)
# os.remove(input_dir+'/cloth_web.jpg')
# os.rename(input_path, input_dir+'/cloth_web.jpg')


# Uncomment the following block if you would like to upload your own images. 

# input_dir = 'static'
# uploaded = files.upload()
# for filename in uploaded.keys():
#   input_path = os.path.join(input_dir, filename)
#   shutil.move(filename, input_path)
# os.remove(input_dir+'/origin_web.jpg')
# os.rename(input_path, input_dir+'/origin_web.jpg')

#%%
import matplotlib.pyplot as plt
import cv2

sample_dir = "./sample_dataset"

sample_model_dir = sample_dir + "/model_0.jpg"
sample_cloth_dir = sample_dir + "/cloth_0.jpg"

# print(cv2.imread(sample_model_dir))

original = cv2.cvtColor(cv2.imread(sample_model_dir), cv2.COLOR_BGR2RGB)
cloth = cv2.cvtColor(cv2.imread(sample_cloth_dir), cv2.COLOR_BGR2RGB)

## Display Images
fig, axes = plt.subplots(nrows=1, ncols=2)
dpi = fig.get_dpi()

fig.set_size_inches(900/ dpi, 448 / dpi)

plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

axes[0].axis('off')
axes[0].imshow(original)
axes[1].axis('off')
axes[1].imshow(cloth)

plt.show()
# %%
