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
result = furhat.listen()

frame_list = []
cont = 0
if result.message == "yes":
    furhat.gesture(name="BigSmile")
    furhat.say(text="That's great, then I will start receiving information.")
    for msg in consumer:
        cont += 1
        bytes = msg.value
        str_res = bytes.decode()
        str_res = str_res.replace("\'", "\"")
        print(str_res)
        js = json.loads(str_res)
        info = str(js['gesture_name'])
        print(info)
        letter = info.split('_')[2]
        frame_list.append(letter)
        if(cont <= 400):
            if len(frame_list) > 1:
                if frame_list[len(frame_list)-1] != frame_list[len(frame_list)-2]:
                    furhat.say(text="The letter that has been recognized is: " + letter)
                    if letter == 'A':
                        furhat.say(text="For example, a word that starts with this letter is: " + "Apple")
                    if letter == 'B':
                        furhat.say(text="For example, a word that starts with this letter is: " + "Boat")
                    if letter == 'C':
                        furhat.say(text="For example, a word that starts with this letter is: " + "Cloud")
                    if letter == 'L':
                        furhat.say(text="For example, a word that starts with this letter is: " + "Laptop")
                    if letter == 'R':
                        furhat.say(text="For example, a word that starts with this letter is: " + "Real")
                    if letter == 'U':
                        furhat.say(text="For example, a word that starts with this letter is: " + "Usual")
            else:
                if letter != '':
                    furhat.say(text="The letter that has been recognized is: " + info.split('_')[2])
                    if letter == 'A':
                        furhat.say(text="For example, a word that starts with this letter is: " + "April")
                    if letter == 'B':
                        furhat.say(text="For example, a word that starts with this letter is: " + "Barcelona")
                    if letter == 'C':
                        furhat.say(text="For example, a word that starts with this letter is: " + "Cold")
                    if letter == 'L':
                        furhat.say(text="For example, a word that starts with this letter is: " + "Letter")
                    if letter == 'R':
                        furhat.say(text="For example, a word that starts with this letter is: " + "Rain")
                    if letter == 'U':
                        furhat.say(text="For example, a word that starts with this letter is: " + "Umbrella")
                else:
                    furhat.say(text="I have not received anything.")
                    furhat.gesture(name="BrowRaise")
        else:
            frame_list.clear()
            cont = 0


else:
   furhat.gesture(name="BrowRaise")
   furhat.say(text="That's fine. I will be here for when you need me.")
   furhat.gesture(name="Smile")
