import flask
import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["DEBUG"] = True

import matplotlib.pyplot as plt
import numpy as np
print("Loading model")
global sess
sess = tf.Session()
set_session(sess)
global model
model = load_model('my_cifar10_model.h5')
global graph




def factors(num):
  return [x for x in range(1, num+1) if num%x==0]

@app.route('/')
def home():
  return "Open the URL and go to /factors/num to see the website"

@app.route('/factors/<int:num>')
def factors_route(num):
  return "The factors of {} are {}".format(num, factors(num))

if __name__ == '__main__':
  app.run(host='0.0.0.0')