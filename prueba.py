from furhat_remote_api import FurhatRemoteAPI
from kafka import KafkaConsumer
import json
import kafka_consumer

consumer = KafkaConsumer('my_topic',group_id='my-group_1',bootstrap_servers=['localhost:9092'])

furhat = FurhatRemoteAPI("localhost")

# Get the voices on the robot
voices = furhat.get_voices()
# Set the voice of the robot
furhat.set_voice(name='Matthew')
# Listen to user speech and return ASR result
result = furhat.listen()
# Attend the user closest to the robot
furhat.attend(user="CLOSEST")
# Attend a user with a specific id
furhat.attend(userid="virtual-user-1")
# Attend a specific location (x,y,z)
furhat.attend(location="0.0,0.2,1.0")

# Set the LED lights
furhat.set_led(red=200, green=50, blue=50)

# Greet the user
furhat.say(text="Hello! I am Furhat, and today I am going to help you to identify letters from the sign language")
furhat.gesture(name="BigSmile")

furhat.say(text="Are you ready to start?", blocking =True)
result=furhat.listen()

if result.message == "yes":
    furhat.gesture(name="BigSmile")
    furhat.say(text="That's great, then I will start receiving information.")
    for msg in consumer:
        bytes = msg.value
        str_res = bytes.decode()
        str_res = str_res.replace("\'", "\"")
        print(str_res)
        js = json.loads(str_res)
        letra = ''
        info = str(js['gesture_name'])
        print(info)
        letter = info.split('_')[2]
        if letter != '':
            furhat.say(text="The letter that has been recognized is letter " + info.split('_')[2])
        else:
            furhat.say(text="I have not received anything.")
            furhat.gesture(name="BrowRaise")
else:
   furhat.gesture(name="BrowRaise")
   furhat.say(text="That's fine. I will be here for when you need me.")
   furhat.gesture(name="Smile")
