'''
import cv2
import time
import smtplib 
import argparse
import schedule

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


# CONSOLE SUPPORT

 
parser = argparse.ArgumentParser(description="Select ticker", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--ticker", help="ticker")
args = parser.parse_args()
config = vars(args)


# DECLARATIONS
# --------------------------------

scheduler_interval_minutes = 5

css = '[aria-label]'

screenshot_before = "main--youtube-screenshot-selenium-before"
screenshot_after = "main--youtube-screenshot-selenium-after"
screenshot_after2 = "main--youtube-screenshot-selenium-after2"
screenshot_main = "main--screenshot"
screenshot_extension = ".png"

folder_database = "storage/database/"
folder_results = "storage/history/"
folder_assets  = "assets/"

ticker = config["ticker"] if ("ticker" in config and type(config["ticker"])== str) else "LUNC-USD"


url = "https://www.youtube.com/watch?v=qqL2TOU-nzk"
urls = {
    "LUNC-USD":"https://www.youtube.com/watch?v=qqL2TOU-nzk",
    "XRP-USD":"https://www.youtube.com/watch?v=AtRjS6uJ4B8",
    "BTC-USD": "https://www.youtube.com/watch?v=Ujaf29NNfwU",
    "SHIB-USD": "https://www.youtube.com/watch?v=UqKC3o-ds6U",
    "CRO-USD":"https://www.youtube.com/watch?v=hgg3L5EpxCM",
    "PEPECOIN-USD":"https://www.youtube.com/watch?v=oxkwyACupvY",
    "FTT-USD":"https://www.youtube.com/watch?v=x-iGtYCB7J0"
}

email_notifications = ["pihh.rocks@gmail.com", "filipemotasa@hotmail.com"]

start_time = time.time()


# HELPERS 
# --------------------------------

# Database
def save_database(ticker,df):
    filename = get_database_filename(ticker)
    df.to_csv(filename,index=False)

def get_database_filename(ticker):
    return folder_database+"db_"+ticker+".csv"

def get_database_file(ticker):
    filename = get_database_filename(ticker)
    try:
        df = pd.read_csv(filename)
        return df
    except:
        df = pd.DataFrame([],["position","date","ticker","price","progress","trade_outcome"])
        save_database(ticker,df)
        return df

def update_dataframe(position,ticker):
    # dict = {'position':[],
    #     'date':[],
    #     'ticker':[],
    #     'price':[],
    #     'progress':[],
    #     'trade_outcome':[]
    #    }

    df = get_database_file(ticker)
    close_price = get_last_price(ticker)
    
    progress = 0
    trade_outcome = 0
    message = "Bot BOUGHT LUNC at the price " + str(close_price) + ' USD.'
    if position == "sell":
        buy_price = df.iloc[-1].price
        trade_outcome = (close_price / buy_price)-1
        progress = (df["trade_outcome"]+1).product()*trade_outcome
        
        message = f"""
        Bot SOLD LUNC at the price {close_price}USD.
        Last purchase was at the price {buy_price}USD.
        
        This transaction has a performance of {trade_outcome}%.
        The total progress of the bot if {progress}% in {int(len(df)/2)} complete transactions.
        
        """
    df.loc[len(df.index)] = [position, get_current_datetime(), ticker,close_price,progress,trade_outcome] 

    save_database(ticker,df)
    
    notify_users(position,message)
    
    
# Dates and time
def get_current_datetime():
    current_dateTime = datetime.now()
    return str(current_dateTime)

def get_parsed_current_datetime():
    current_dateTime = get_current_datetime()
    return str(current_dateTime).replace(' ','_').replace('.','__').replace(':','-')

def elapsed_time_log(message):
    st.text("["+message+"] %s seconds" % (time.time() - start_time))

# Screenshots
def get_screenshot_file_name(name,ticker,hasDate=True):
    if hasDate:
        name = name +'__'+get_parsed_current_datetime()
    return folder_results+ticker+'__'+name+screenshot_extension

def get_asset_file_name(name,hasDate=True):
    if hasDate:
        name = name +'__'+get_parsed_current_datetime()
    return folder_assets+name+screenshot_extension

# Trader
def get_last_price(ticker):
    prices =yf.download(tickers=ticker, period="1d", interval="1m")
    return prices.iloc[-1].Close

def get_position(ticker):
    df = get_database_file(ticker)
    df_len = len(df)
    if(df_len == 0):
        return "buy"
    else:
        return "sell" if df.iloc[-1].position == "buy" else "buy"
    
# Computer vision
def find_icon_in_screen(icon,ticker):
    img_rgb = cv2.imread(get_screenshot_file_name(screenshot_main,ticker,False))
    template = cv2.imread(get_asset_file_name(icon,False))
    
    w, h = template.shape[:-1]

    res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    threshold = .8
    loc = np.where(res >= threshold)

    maxX = 0
    for pt in zip(*loc[::-1]):  # Switch columns and rows
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    
        if pt[0]> maxX:
            maxX = pt[0]
       

    cv2.imwrite('main_result_'+icon+'.png', img_rgb)
    return maxX
    

# Notifications
def notify_users(action, message):

    sender = "pihh.rocks@gmail.com"
    recipients = email_notifications
    password = "jtqb rrub hsma pdkv"
    
    
    subject = "[LunaBot Action]::"+action
    body = message


    def send_email(subject, body, sender, recipients, password):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(sender, password)
            smtp_server.sendmail(sender, recipients, msg.as_string())
        st.text("Message sent!")


    send_email(subject, body, sender, recipients, password)


def check_for_signals(driver,ticker):
    
    #df = get_database_file(ticker)
    position = get_position(ticker)
    driver.save_screenshot(get_screenshot_file_name(screenshot_main,ticker))
    driver.save_screenshot(get_screenshot_file_name(screenshot_main,ticker,False))  # change image name
    buyX = find_icon_in_screen('buy',ticker)
    sellX =find_icon_in_screen('sell',ticker)
  
    if(sellX > buyX):
        if position =="sell":
            st.text('Sell', get_current_datetime() )
            update_dataframe(position,ticker)
            position = "buy"
            
    if(buyX > sellX  ):
        if position == "buy":
            st.text('Buy', get_current_datetime() )
            update_dataframe(position,ticker)
            position = "sell"
            




# MAIN FUNCTIONALITY
# --------------------------------   


def main(ticker):
    url = urls[ticker]

    # DRIVER SETUP
    # --------------------------------   
    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    options.add_argument("--headless")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(options=options)

    #driver = webdriver.Chrome(
    #    options=options, executable_path="C:\drivers\chromedriver.exe")

    stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )

    # Start driver
    elapsed_time_log("Driver get")
    driver.get(url)

    # Take first screenshot
    wait = WebDriverWait(driver, 1000)
    driver.save_screenshot(get_screenshot_file_name(screenshot_before,ticker,False))  # change image name
    elapsed_time_log("Screenshot before")

    # Find element to click
    elements = driver.find_elements(By.CSS_SELECTOR,css)
    for el in elements:
        if 'Accept all' in el.get_attribute('innerText'):
            el.click()

    # Take screenshot after click
    wait = WebDriverWait(driver, 5000)       
    driver.save_screenshot(get_screenshot_file_name(screenshot_after,ticker,False))  # change image name
    elapsed_time_log("Screenshot after")

    # Take final screenshot
    wait = WebDriverWait(driver, 30000)
    elapsed_time_log("Screenshot final")

    # Setup complete
    driver.save_screenshot(get_screenshot_file_name(screenshot_after2,ticker,False)) 
    elapsed_time_log("Driver setup complete")


    # RUN LOOP
    # --------------------------------   
    check_for_signals(driver,ticker)

    def task():
    # Perform the desired operations here
        st.text("Check for signal updates...")
        check_for_signals(driver,ticker)

    # Schedule the task to run every 5 minutes
    schedule.every(scheduler_interval_minutes).minutes.do(task)

    # Infinite loop schedule
    while True:
        schedule.run_pending()
        time.sleep(1)

main(ticker)
'''

import streamlit as st

"""
## Web scraping on Streamlit Cloud with Selenium

[![Source](https://img.shields.io/badge/View-Source-<COLOR>.svg)](https://github.com/snehankekre/streamlit-selenium-chrome/)

This is a minimal, reproducible example of how to scrape the web with Selenium and Chrome on Streamlit's Community Cloud.

Fork this repo, and edit `/streamlit_app.py` to customize this app to your heart's desire. :heart:
"""

with st.echo():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    #@st.cache_resource 
    def get_driver(options):
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--headless')

    driver = get_driver(options)
    driver.get("https://google.com")

    st.code(driver.page_source)