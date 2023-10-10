from PIL import Image
import glob
import math
import os


class Collage():
    def make(self, input_folder, output_file, width):
        input_images = glob.glob(os.path.join(input_folder, '*.bmp'))
        if len(input_images) < 1:
            print (f"[-] There are no images in {input_folder}.")
            return
        input_images.sort()
        h, w = Image.open(input_images[0]).convert('RGB').size
        # create folder if it doesn't exist
        if not os.path.exists(os.path.dirname(output_file)):
            os.mkdir(os.path.dirname(output_file))
        self.collage_maker(input_images, output_file, width*w)
        print (f"Collage for {input_folder} is stored as {output_file}.")


    def collage_maker(self, images, filename, width = 1024, init_height = 64):
        """
        Collage maker - tool to create picture collages
        Source: https://github.com/delimitry/collage_maker.git

        Make a collage image with a width equal to `width` from `images` and save to `filename`.
        """
        if not images:
            print('No images for collage found!')
            return False

        margin_size = 2
        # run until a suitable arrangement of images is found
        while True:
            # copy images to images_list
            images_list = images[:]
            coefs_lines = []
            images_line = []
            x = 0
            while images_list:
                # get first image and resize to `init_height`
                img_path = images_list.pop(0)
                img = Image.open(img_path)
                img.thumbnail((width, init_height))
                # when `x` will go beyond the `width`, start the next line
                if x > width:
                    coefs_lines.append((float(x) / width, images_line))
                    images_line = []
                    x = 0
                x += img.size[0] + margin_size
                images_line.append(img_path)
            # finally add the last line with images
            coefs_lines.append((float(x) / width, images_line))

            # compact the lines, by reducing the `init_height`, if any with one or less images
            if len(coefs_lines) <= 1:
                break
            if any(map(lambda c: len(c[1]) <= 1, coefs_lines)):
                # reduce `init_height`
                init_height -= 10
            else:
                break

        # get output height
        out_height = 0
        for coef, imgs_line in coefs_lines:
            if imgs_line:
                out_height += int(init_height / coef) + margin_size
        if not out_height:
            print('Height of collage could not be 0!')
            return False

        collage_image = Image.new('RGB', (width, int(out_height)), (35, 35, 35))
        # put images to the collage
        y = 0
        for coef, imgs_line in coefs_lines:
            if imgs_line:
                x = 0
                for img_path in imgs_line:
                    img = Image.open(img_path)
                    # if need to enlarge an image - use `resize`, otherwise use `thumbnail`, it's faster
                    k = (init_height / coef) / img.size[1]
                    if k > 1:
                        # maintain the last row images' dimensions to be the same as the top rows
                        #img = img.resize((int(img.size[0] * k), int(img.size[1] * k)), Image.LANCZOS)
                        img.thumbnail((int(width / coef), int(init_height / coef)), Image.LANCZOS)
                    else:
                        img.thumbnail((int(width / coef), int(init_height / coef)), Image.LANCZOS)
                    if collage_image:
                        collage_image.paste(img, (int(x), int(y)))
                    x += img.size[0] + margin_size
                y += int(init_height / coef) + margin_size
        collage_image.save(filename)
        return True


