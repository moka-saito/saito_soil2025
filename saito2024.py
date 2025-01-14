import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import ParticleSegmenter
from visualization import vis_segmap


#%%　画像の読み込み

image = cv2.imread('/Users/saitomoka/my_project/bpartis/segment/symposium2024/gray05.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

#%%　モデル読み込み
model_path = '/Users/saitomoka/my_project/bpartis/models/seg-model.pt'
model_path = os.path.abspath(model_path)
segmenter = ParticleSegmenter(bayesian=False, model_path=model_path, device='cpu')


#segmenter = ParticleSegmenter(model_path='/Users/saitomoka/my_project/bpartis/models/seg-model.pt', device='cpu')
segmentation, uncertainty, _ = segmenter.segment(image)

os.makedirs('./results', exist_ok=True)
cv2.imwrite('./results/gray_seg.png', segmentation)

seg_cl, seg_cl_concat = vis_segmap(segmentation, image=image)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(2,1,1)
ax.imshow(seg_cl)
ax = fig.add_subplot(2,1,2)
ax.imshow(seg_cl_concat)
plt.show()



#%%
print('shape:',segmentation.shape)
print('labels:', np.unique(segmentation))



#%%
# ラベルごとの粒子の面積と最小外接円の直径を計算
particle_areas = {}
particle_diameters = {}

labels = np.unique(segmentation)
print('粒子ラベル:', labels)

for label in labels:
    # 粒子のマスクを作成（指定ラベルに対応する部分が1、それ以外が0）
    mask = (segmentation == label).astype(np.uint8)

    # 輪郭を抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # 最初の輪郭に対して最小外接円を計算
        (x, y), radius = cv2.minEnclosingCircle(contours[0])
        diameter = 2 * radius  # 直径 = 半径 × 2
        
        # 面積を計算（ラベルの画素数をカウント）
        area = np.sum(mask)
        
        # 結果を辞書に保存
        particle_areas[label] = area
        particle_diameters[label] = diameter

# 結果の表示
print('各粒子の面積:', particle_areas)
print('各粒子の最小外接円の直径:', particle_diameters)

# 粒子の総数、平均面積、平均直径を表示
total_particles = len(particle_areas)
average_area = np.mean(list(particle_areas.values())) if total_particles > 0 else 0
average_diameter = np.mean(list(particle_diameters.values())) if total_particles > 0 else 0

print(f'総粒子数: {total_particles}')
print(f'平均粒子面積: {average_area:.2f} ピクセル')
print(f'平均粒子直径: {average_diameter:.2f} ピクセル')

import csv

# 保存先のファイル名
csv_filename = '/Users/saitomoka/my_project/bpartis/segment/symposium2024/result/plots05/particle_analysis05.csv'

# ヘッダーとデータの準備
header = ['Particle Label', 'Area (pixels)', 'Diameter (pixels)']
rows = []

for label in particle_areas:
    area = particle_areas[label]
    diameter = particle_diameters[label]
    rows.append([label, area, diameter])

# CSVファイルに書き込み
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # ヘッダーを書き込み
    writer.writerows(rows)   # 各粒子のデータを書き込み

print(f'粒子の分析結果が {csv_filename} に保存されました')

#%%
def overlay_labels_on_image(original_img, segmented_img, labels):
    # 元の画像をカラーで読み込む（グレースケールの場合はカラーマップを適用）
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    # 各ラベルの重心を求めて画像上に番号を表示
    for label_id in np.unique(labels):
        if label_id == 0:  # 背景のラベル（通常0）を無視
            continue
        
        # ラベルに対応するマスクを作成
        mask = (labels == label_id).astype(np.uint8)
        
        # 各ラベルの重心を計算
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # 元の画像にラベル番号を重ねる
            cv2.putText(original_img, str(label_id), (cX, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return original_img


# 元の入力画像とラベル画像を読み込む（ここでは例としてランダムに生成）
original_img = cv2.imread('/Users/saitomoka/my_project/bpartis/segment/symposium2024/gray05.jpg')
labels = cv2.imread('/Users/saitomoka/my_project/bpartis/segment/results/gray_seg.png', 0)  # ラベル画像はグレースケール

# ラベルを重ねた画像を生成
labeled_image = overlay_labels_on_image(original_img, labels, labels)

# Matplotlibで表示
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Labeled Particles')
plt.show()



#%%
def overlay_specific_labels_on_image(original_img, labels, target_labels):
    # 元の画像をカラーで読み込む（グレースケールの場合はカラーマップを適用）
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    # 指定したラベル番号の粒子のみを処理
    for label_id in target_labels:
        # ラベルに対応するマスクを作成
        mask = (labels == label_id).astype(np.uint8)
        
        # 各ラベルの重心を計算
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # 元の画像にラベル番号を重ねる
            cv2.putText(original_img, str(label_id), (cX, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return original_img


# 元の入力画像とラベル画像を読み込む
original_img = cv2.imread('original_image.jpg')
labels = cv2.imread('segmentation_labels.png', 0)  # ラベル画像はグレースケール

# 表示したい特定のラベル番号を指定（例: 35と36）
target_labels = [32, 36]

# 特定のラベルのみを重ねた画像を生成
labeled_image = overlay_specific_labels_on_image(original_img, labels, target_labels)

# Matplotlibで表示
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Labeled Particles for Specific Labels')
plt.show()











#%%


fig, ax = plt.subplots()
im = ax.imshow(uncertainty)
fig.colorbar(im, ax=ax)
plt.show()


#%%
from measurement import Measurer

image = cv2.imread('/Users/saitomoka/my_project/bpartis/segment/symposium2024/gray02.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

segmentation = cv2.imread('./results/gray_seg.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(segmentation)
plt.show()


#%%
image = cv2.imread('/Users/saitomoka/Desktop/saito2024/20241113/Wed Nov 13 15-46-05.jpg')

measurer = Measurer(segmentation, image)
print(measurer.params)

# CSVファイルに保存する
measurer.params.to_csv('/Users/saitomoka/my_project/bpartis/segment/symposium2024/result/measurer_params_02.csv', index=True)




plt.imshow(measurer.draw_contours())
plt.show()


img = measurer.draw_contours()
plt.imshow(measurer.draw_num(img))
plt.show()



# 最小外接短形
plt.imshow(measurer.draw_rect())
plt.show()

# 最小外接円
plt.imshow(measurer.draw_circle())
plt.show()

# 楕円近似
plt.imshow(measurer.draw_ellipse())
plt.show()






























