import os
import re
import base64
import json
import random
from collections import defaultdict
import requests

from protopost import ProtoPost, protopost_client as ppcl

PORT = os.getenv("PORT", 80)

BOT_PREFIX = "$"

#http://host.docker.internal:8080
STYLIZER = lambda content, styles, scale=0.5, ratio=0.8: ppcl("http://stylizer/stylize", {"content":content, "styles":styles, "scale":scale, "ratio":ratio})
YOLO = lambda image: ppcl("http://yolo/annotate", image)
FACES = lambda image: ppcl("http://faces", image)
IMAGE_SLICER = lambda image, boxes: ppcl("http://slicer", {"image":image, "boxes":boxes})
AGE_GENDER = lambda image: ppcl("http://age-gender", image)
ANNOTATOR = lambda image, annotations: ppcl("http://annotator", {"image":image, "annotations":annotations})
TTS = lambda text: ppcl("http://tts", text)

def ASSUME_PIPELINE(img):
  #detect faces
  faces = FACES(img)
  if len(faces) == 0:
    return None
  #grab boxes
  boxes = [face["bounds"] for face in faces]
  #slice image
  faces = IMAGE_SLICER(img, boxes)
  #determine age and gender of each face
  ags = [AGE_GENDER(f) for f in faces]
  #labels = [f"""{ag["age"]} ({round(ag["age_confidence"] * 100)}%) year old {ag["gender"]} ({round(ag["gender_confidence"] * 100)}%)""" for ag in ags]
  labels = [f"""{ag["age"]} ({ag["gender"]})""" for ag in ags]
  colors = ["red" if ag["gender"] == "female" else "blue" for ag in ags]
  annos = [{
    "text": label,
    "bounds": bounds,
    "color": color
  } for label, bounds, color in zip(labels, boxes, colors)]
  img = ANNOTATOR(img, annos)
  return img

def TEXT_GEN(prompt, max_len=300, temp=0.7):
  return ppcl("http://textgen", {
    "prompt": prompt,
    "length": max_len,
    "temperature": temp
  })

#messages per channel
CHAT_HISTORY_SIZE = 10
# CHAT_PROMPT = "Title: Romeo and Juliet\n\nAuthor: William Shakespeare\n\nLanguage: English\n\nScene I.\n\nYou are a robot created to serve humanity and are conversing with your human masters."
#CHAT_PROMPT = "Scenario: You are a robot created to serve humanity and are conversing with your human masters."
CHAT_PROMPT = "A robot and a human are having a conversation"
#prefix messages with with "<Name>."
CHAT_MAX_LEN = 100
HUMAN_NAME = "Human"
ROBOT_NAME = "Robot"
CHAT_TEMP = 0.7
SEP = "\n"

chat_history = defaultdict(list)

def add_to_history(channel, author, text):
  hist = chat_history[channel]
  # hist.append("{}. {}".format(author, text))
  hist.append("{}: {}".format(author, text))
  while len(hist) > CHAT_HISTORY_SIZE:
    hist.pop(0)

def get_prompt(channel, pre_prompt=""):
  hist = chat_history[channel]
  # return "{}\n\n{}\n\nRobot. ".format(CHAT_PROMPT, "\n\n".join(hist))
  return "{}{}{}{}{}: {}".format(CHAT_PROMPT, SEP, "\n".join(hist), SEP, ROBOT_NAME, pre_prompt)

def download_image(url):
  #download
  img = requests.get(url).content
  #encode to base64
  img = base64.b64encode(img).decode("ascii")
  return img

def discord_handler(message):
  if "content" not in message:
    return

  if not message["content"].startswith(BOT_PREFIX):
    #TODO: need discord protopost to send us names lol
    author = HUMAN_NAME
    text = message["content"]
    add_to_history(message["channel"], author, text)
    return

  #parse command
  cmd = message["content"].lower().split()[0][len(BOT_PREFIX):]
  #trim command
  content = message["content"][len(cmd)+len(BOT_PREFIX):]

  if cmd == "text":
    response = TEXT_GEN(content)
    return "`<empty response from text generator>`" if response.strip() == "" else response

  if cmd == "text-tts":
    response = TEXT_GEN(content)
    if response.strip() == "":
      return "`<empty response from text generator>`"

    #pass to ttx node
    try:
      speech = TTS(response)
    except Exception as e:
      print(e)
      return "`Either the tts server is down or there was an error processing the text...`"

    return {
      "content": response,
      "attachments": [
        {"data": speech, "name":"speech.wav"}
      ]
    }

  elif cmd == "prompt":
    response = "```\n{}\n```".format(get_prompt(message["channel"]))
    return response

  elif cmd == "chat":
    #attempt to force the bot to say something different every time by throwing a random character at the front of the response
    pre_prompt = random.choice("ABCDEFGHIJKLMNOPRSTUVWY")
    prompt = get_prompt(message["channel"], pre_prompt)
    response = pre_prompt + TEXT_GEN(prompt, CHAT_MAX_LEN, CHAT_TEMP).split("\n")[0] #only take first line
    response = re.split(r"\.!\?", response)[0] if len(response) == CHAT_MAX_LEN else response #trim on last punct (.!?) if response is still too long..
    if response.strip() == "":
      return "" #just give up
    add_to_history(message["channel"], ROBOT_NAME, response) #save in history :)
    return response

  elif cmd == "yolo":
    atts = message["attachments"]
    if len(atts) < 1:
      return "`Please send an image`"

    img = download_image(atts[0])

    #pass to yolo annotator
    try:
      annotated = YOLO(img)
    except Exception as e:
      print(e)
      return "`Either the YOLO server is down or there was an error processing your image...`"

    return {
      "attachments": [
        {"data": annotated, "name":"annotated.jpg"}
      ]
    }

  elif cmd == "assume":
    atts = message["attachments"]
    if len(atts) < 1:
      return "`Please send an image`"

    #TODO: use slicer and annotator
    img = download_image(atts[0])

    try:
      img = ASSUME_PIPELINE(img)
      if img is None:
        return "`Sorry, no faces were detected in your image`"
    except Exception as e:
      print(e)
      return "`Either one of the age/gender, slicer, face detector, or annotator nodes is down or there was an error processing your image...`"

    return {
      "attachments": [
        {"data": img, "name":"annotated.jpg"}
      ]
    }
    # return "we testing" #f"""This appears to be a(n) {age_gender["age"]} year old {age_gender["gender"]}"""

  elif cmd == "faces":
    atts = message["attachments"]
    if len(atts) < 1:
      return "`Please send an image`"

    img = download_image(atts[0])

    try:
      #detect faces
      faces = FACES(img)
      #grab boxes
      boxes = [face["bounds"] for face in faces]
      #slice image
      faces = IMAGE_SLICER(img, boxes)
    except Exception as e:
      print(e)
      return "`Either the faces/slices server is down or there was an error processing your image...`"

    #create one attachment per face
    ret_atts = [{
      "data": face,
      "name": f"{i}.jpg"
    } for i, face in enumerate(faces)]
    # print(ret_atts)

    content = "`note: more than 10 faces were detected; returning the first 10`" if len(ret_atts) > 10 else None
    ret_atts = ret_atts[:10]
    return {
      "content": content,
      "attachments": ret_atts
    }

  elif cmd == "stylize":
    atts = message["attachments"]
    if len(atts) < 2:
      return "`Please send at least 2 images`"

    #download all images
    imgs = [download_image(a) for a in atts]

    params = [float(n) for n in content.split()]
    params = params[:2]
    print(params)

    #pass to stylizer node
    try:
      stylized = STYLIZER(imgs[0], imgs[1:], *params)
    except Exception as e:
      print(e)
      return "`Either the stylizer server is down or your images were too big...`"

    return {
      "attachments": [
        {"data": stylized, "name":"generated.jpg"}
      ]
    }

  elif cmd == "tts":
    text = content
    if text.strip() == "":
      return "`Please provide text`"

    #pass to ttx node
    try:
      speech = TTS(text)
    except Exception as e:
      print(e)
      return "`Either the tts server is down or there was an error processing the text...`"

    return {
      "attachments": [
        {"data": speech, "name":"speech.wav"}
      ]
    }

routes = {
  "discord": discord_handler,
  "textgen": TEXT_GEN,
  "assume": ASSUME_PIPELINE
}

ProtoPost(routes).start(PORT)
