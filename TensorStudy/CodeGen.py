from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random
import sys
 
number = ['0','1','2','3','4','5','6','7','8','9']
# alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def random_captcha_text(char_set=number, captcha_size=4):

    captcha_text = []
    for i in range(captcha_size):

        c = random.choice(char_set)

        captcha_text.append(c)
    return captcha_text
 

def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text()

    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    image.write(captcha_text, 'captcha/images/' + captcha_text + '.jpg')  # д���ļ�


num = 10000
if __name__ == '__main__':
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write('\r>> Creating image %d/%d' % (i+1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
                        
    print("111")