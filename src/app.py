from flask import Flask, render_template, url_for, request, redirect
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config["IMAGE"] =  "C:/Users/willy/Tubes-Algeo_2_IF_2123/image"
app.config["ALLOW"] =["PNG", "JPG", "JPEG", "GIF"]

def allowed(filename):
   if not "." in filename:
      return False

   format = filename.rsplit(".", 1)[1]
   if format.upper() in app.config["ALLOW"]:
      return True
   else:
      return False

@app.route('/', methods=["GET","POST"])
def main():
   if request.method == "POST":
      if request.files:
         image = request.files["image"]
         if not(allowed(image.filename)):
            print("Image extension is not allowed")
         else:
            filename = secure_filename(image.filename)
            if not(os.path.isdir(app.config["IMAGE"])):
               os.makedirs(app.config["IMAGE"])
            image.save(os.path.join(app.config["IMAGE"], filename))
            print("Image saved")
         return redirect(request.url)
   return render_template('gwf.html')

if __name__ == '__main__':
   app.run(debug=True)