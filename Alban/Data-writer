from tkinter import *
import PIL
from PIL import Image, ImageDraw

#this one is for data entry and saves them as 28x28 pixel images directly on our desired data folder
def save():
    global image_number
    filename = f'image_{image_number}.png'   # image_number increments by 1 at every save
    image1.save(filename)
    image_number += 1


def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=1)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='black', width=2)
    lastx, lasty = x, y


root = Tk()

lastx, lasty = None, None
image_number = 0

cv = Canvas(root, width=28, height=28, bg='white')
# --- PIL
image1 = PIL.Image.new('RGB', (28, 28), 'white')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="save", command=save)
btn_save.pack()

root.mainloop()
