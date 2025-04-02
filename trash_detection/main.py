
from flask import Flask,
from predict import predict_main
from points import User

def main():
    user = User()
    final_label = predict_main(img)
    user.detect_trash(final_label)
    return final_label