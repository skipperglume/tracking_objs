import cv2
from PIL import Image
from ultralytics import YOLO


model = YOLO("models/yolo26s.pt")

# models/yolo26n.pt
# models/yolo26s.pt


# from PIL
im1 = Image.open("Inputs/bridge_cars.jpg")
results = model.predict(source=im1, save=True, show=True)  # save plotted images


# Print the im1 dimensions:
print(f"Image 1 dimensions: {im1.size}")


print(type(results))
print(len(results))
print(dir(results[0]))
print(results[0])

print("-" * 38)
print("boxes")
print(results[0].boxes)
print("-" * 38)

# print("boxes.xyxy")
# print(results[0].obb)
# print("probs")
# print(results[0].probs)
# print("names")
# print(results[0].show)

# Save resultd from running on video "Inputs/video.mp4":
# results_video = model.predict(
#     source="Inputs/video.mp4", save=True, show=True, stream=True
# )
# # results_video

# print(f"Processed {len(results_video)} frames from video.")


# apply', '_keys', 'boxes', 'cpu', 'cuda', 'keypoints', 'masks', 'names', 'new', 'numpy', 'obb', 'orig_img', 'orig_shape', 'path', 'plot', 'probs', 'save', 'save_crop', 'save_dir', 'save_txt', 'show', 'speed', 'summary', 'to', 'to_csv', 'to_df', 'to_json', 'update', 'verbose']
# print("Raw results from model:")
# print(results)
