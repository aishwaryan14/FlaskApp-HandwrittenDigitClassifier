from flask import Flask,redirect,render_template,request,url_for,session
import numpy as np
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import pickle


app=Flask(__name__)
app.secret_key="secret"
@app.route("/",methods=['GET','POST'])

def index():
	try:
		prediction = session["prediction"]
		accuracy = session["accuracy"]
	except KeyError:
		prediction = session["prediction"] = ""
		accuracy = session["accuracy"] = ""
	if request.method=='GET':
		return render_template('index.html',prediction="",accuracy="")
	if request.method=='POST':
		if request.form["action"]=="SUBMIT":
			pred=request.form["pic"]
			img = Image.open("static/"+pred)
			img1 = img.resize((8,8))
			img1= img1.convert('L')
			img1 = ImageOps.invert(img1)
			temp=np.asarray(img1)/255
			temp=temp.flatten()
			loaded_model = pickle.load(open("model.pkl","rb"))
			result = loaded_model.predict([temp])
			print(result[0])
			accuracy = round(max(loaded_model.predict_proba([temp])[0])*100,2)
			plt.matshow(img1)
			#plt.imshow(img1)
			#plt.savefig('static/invimg.png')
			plt.show()
			return render_template('index.html',prediction=result[0],predict="Prediction:",accuracy=accuracy,Accuracy="Accuracy:")

if __name__ == '__main__':
	app.run(host='localhost', port=1098, debug=True)