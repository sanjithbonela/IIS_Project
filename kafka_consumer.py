from kafka import KafkaConsumer
import json
if __name__ == '__main__':
    consumer = KafkaConsumer('my_topic',group_id='my-group_1',bootstrap_servers=['localhost:9092'])
    for msg in consumer:
        #run e2e_test.test_video de github y los dos producers y cambiar esto de consumer y ya esta
        #convertir msg bytes a string y luego json
        bytes= msg.value
        str = bytes.decode()
        js= json.loads(str)
        letra = ''
        info =js['gesture_name']
        info = "ASL_letter_" + letra


def myFunction():
    return letra





