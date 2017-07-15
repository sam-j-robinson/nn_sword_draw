import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

input_dir = os.getcwd()+'/input/'

image_loc = input_dir + 'zz_all_cut/'

number_of_swords = 5
parts = ['train', 'blade', 'grip', 'magic']
blade_flip = False

def return_sword_img_names(sword_num, part, hue_shift=False):
    if hue_shift:
        return 'sword_%s_%s_%s.png' %(sword_num, part, hue_shift)
    else:
        return 'sword_%s_%s.png' %(sword_num, part)

def hue_shift(img, amount):
    hsv_img = img.convert('HSV')
    hsv = np.array(hsv_img)
    hsv[..., 0] = (hsv[..., 0]+amount) % 360
    new_img = Image.fromarray(hsv, 'HSV')
    return new_img.convert('RGB')

def alpha_hue_shift(img_png, img_rgb, shift_amount):
    final_rgb = hue_shift(img_rgb, shift_amount)
    
    old_r, old_g, old_b, final_a = img_png.split()
    fin_r, fin_g, fin_b = final_rgb.split()

    png_np = np.dstack((fin_r, fin_g, fin_b, final_a))
    img_alpha_final = Image.fromarray(png_np, 'RGBA')
    return img_alpha_final

def main():
    for i in range(1, number_of_swords+1):
        print("Sword: %s" %i)
        for part in parts:
            print("Parts: %s" %part)
            input_name = return_sword_img_names(i, part)
            input_loc = image_loc + input_name

            img_png = Image.open(input_loc)
            img_rgb = img_png.convert('RGB')

            for shift in tqdm(range(1, 361, 1)):
                output_name = return_sword_img_names(i, part, shift)
                shift_img = alpha_hue_shift(img_png, img_rgb, shift)
                output_loc = '%ssword_%s/%s' %(input_dir, part, output_name)
                shift_img.save(output_loc, 'PNG')

            img_png.close()


if __name__ == "__main__":
    main()

