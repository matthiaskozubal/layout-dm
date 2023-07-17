from PIL import ImageFont, ImageDraw

def draw_text_in_bbox(bbox, font_path, text, combined_image, text_color = (255, 255, 255)):
    # input
    center_x, center_y, bbox_width, bbox_height = bbox
    top_left_x = center_x - bbox_width//2
    top_left_y = center_y - bbox_height//2
    
    # iterate with increasing font size to find the one that fits the bbox
    font_size = 1
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = font.getsize(text)    
    while text_width < bbox_width and text_height < bbox_height:
        font_size += 1
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = font.getsize(text)
    
    # draw
    draw = ImageDraw.Draw(combined_image)
    draw.text((top_left_x, top_left_y), text, font=font, fill=text_color)
    