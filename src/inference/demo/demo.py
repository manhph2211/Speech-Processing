from flask import Flask, render_template,request
import os
import sys

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


if __name__=="__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4426)))