import os
from PIL import Image, ImageDraw, ImageFont
from trainer.global_configs import FONTS_DIR, OUTPUT_DIR, CANVAS_WIDTH, CANVAS_LENGTH

def combine_elements_based_on_layout_dm(predicted_layout_pages, output_path=OUTPUT_DIR, font_path=f'{FONTS_DIR}/KohSantepheap-Regular.ttf', verbatim=False):
    """
    predicted_layouts: {"testheader.header":(40, 50,500,300),"backgroundo.png":(100, 100, 200, 50),
    "iamtext.txt":(200, 200, 100, 100),
    "Cosmetic_Logo.png":(200, 200, 100, 100), 
    "Cosmetic_Logo copy.png":(10, 50, 100, 100)},
    output_path: "./here_my_output_image.png"
    """

    n_pages = len(predicted_layout_pages)
    for page in range(n_pages):
        # input for this function
        predicted_layouts = predicted_layout_pages[page]
                
        # create an empty list to store the inputs objects and coordinates
        image_list = []
        coordinates_list = []
        text_list = []
        header_list = []
        
        # run over dico of items
        for file_path, coordinates in predicted_layouts.items():
            try:
                #handeling images
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    #open the image file using PIL
                    image = Image.open(file_path)
                    # resize
                    if (image.size[0] > CANVAS_WIDTH) or (image.size[0] > CANVAS_LENGTH):
                        image.thumbnail((CANVAS_WIDTH, CANVAS_LENGTH), Image.ANTIALIAS)
                    # convert image to RGBA mode
                    image = image.convert("RGBA")
                    # append the image object and coordinates to the respective lists
                    image_list.append(image)
                    coordinates_list.append(coordinates)
                # handeling texts
                elif file_path.lower().endswith(('.txt')):
                    # read text file contents
                    with open(file_path, 'r') as file:
                        text = file.read()
                    # append the text and coordinates to the respective lists
                    text_list.append((text, coordinates))

                # handeling headers
                elif file_path.lower().endswith(('.header')):
                    # read text file contents
                    with open(file_path, 'r') as file:
                        header = file.read()
                    # append the header and coordinates to the respective lists
                    header_list.append((header, coordinates))
            except IOError:
                print("Unable to open file:", file_path)
        
        # get the dimensions of the first image
        width, height = image_list[0].size
        
        # create a new blank image with the combined size
        combined_image = Image.new("RGBA", (width, height))
        
        # iterate though the image list and paste each image onto the combined image using the corresponding coordinates
        for image, coordinates in zip(image_list, coordinates_list):
            shifted_coordinates = (coordinates[0] - coordinates[2]//2, coordinates[1] - coordinates[3]//2)
            combined_image.paste(image, shifted_coordinates, image)
        
        # generate a text layer
        draw = ImageDraw.Draw(combined_image)
        for text, coordinates in text_list:
            text_position = (coordinates[0], coordinates[1])
            text_color = (255, 255, 255)  # Colot for the text (set to white)
            font = ImageFont.truetype(font_path, 20)  # font type and size for text
            draw.text(text_position, text, fill=text_color, font=font)

        # generate a text layer
        draw = ImageDraw.Draw(combined_image)
        for header, coordinates in header_list:
            header_position = (coordinates[0], coordinates[1])
            text_color = (255, 255, 255)  # color for the header (set to white)
            font = ImageFont.truetype(font_path, 100)  # font type and size for header
            draw.text(header_position, header, fill=text_color, font=font)
        
        # save the combined image to the specified output path
        output_path_name = os.path.join(output_path, f'combined-images_option-{page}.png')
        combined_image.save(output_path_name)
        ## print
        if verbatim:
            input_print = {os.path.basename(key): val for key, val in predicted_layouts.items()}
            print(f"\nPage {page}\nInput: {input_print}\nCombined image saved to: {output_path_name}\n")


