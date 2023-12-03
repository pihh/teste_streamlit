import cv2
import time
import schedule
import smtplib
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from email.mime.text import MIMEText


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium_stealth import stealth

from datetime import datetime


st.title("OpenCV Demo App")
st.subheader("This app allows you to play with Image filters!")
st.text("We use OpenCV and Streamlit for this demo")