from flask import Flask 

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "./project/uploads"
app.config['SECRET_KEY']='d4552cab351cd4b708cbbd3d26ec2f04'


from project import routes