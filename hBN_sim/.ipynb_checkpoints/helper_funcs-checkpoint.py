#from datetime import datetime

def timeformat():
    '''creates timestamp of format YYYYMMDDHHMMSS'''
    #get current time
    now = datetime.now()
    now_str = [str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute), str(now.second),]
    
    #make sure date has same format every time: YYYYMMDDHHMMSS
    for i in range(len(now_str)):
        if len(now_str[i]) == 1:
            now_str[i] = '0' + now_str[i]
    now_str = ''.join(now_str)
    
    return(now_str)