import io
import numpy as np
import telebot
from telebot import types
import PIL.Image as Image
import cv2
from detector import ATSSHumanFaceAssocDetector, draw_associations


TOKEN = 
bot = telebot.TeleBot(TOKEN)
detector = ATSSHumanFaceAssocDetector(use_gpu=False, fp16=False)


def get_image_handler(img_arr):
    ret, img_encode = cv2.imencode('.jpg', img_arr[..., ::-1])
    str_encode = img_encode.tobytes()
    img_byteio = io.BytesIO(str_encode)
    img_byteio.name = 'img.jpg'
    reader = io.BufferedReader(img_byteio)
    return reader

def get_predict(img):
    human_boxes, face_boxes, association = detector.predict(img)
    img = draw_associations(img, human_boxes, face_boxes, association)

    return img

@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

    bot.send_message(message.chat.id, f'{message.from_user.first_name}, отправьте сюда фото с людьми. Бот вернет изображение с детекцией лиц, тел и ассоциаций между ними.', reply_markup=markup)

@bot.message_handler(content_types=['photo'])
def get_photo(message):
    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    bot.reply_to(message, 'Благодарим Вас за ваше обращение, ваша фото скоро будет обработано')

    image = np.array(Image.open(io.BytesIO(downloaded_file)))
    asoccs_image = get_predict(image)
    returner = get_image_handler(np.array(asoccs_image))
    bot.send_photo(message.chat.id, returner)

bot.polling(non_stop=True)