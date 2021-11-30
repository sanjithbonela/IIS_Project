import kafka
from furhat_remote_api import FurhatRemoteAPI
import rest
from src import kafka_consumer

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
furhat.listen_stop
if result.message == "Yes":
    furhat.gesture(name="BigSmile")
    furhat.say(text="That's great, then I will start receiving information.")
    letra_recon = ''
    kafka_consumer.my_function(letra_recon)

    if letra_recon != '':
        furhat.say(text="The letter that has been recognized is letter" + letra_recon)

else:
    furhat.gesture(name="BrowRaise")
    furhat.say(text="That's fine. I will be here for when you need me.")
    furhat.gesture(name="Smile")
