from kafka import KafkaProducer
from gr_test import *

if __name__ == '__main__':
    prod = KafkaProducer(bootstrap_servers=['localhost:9092'])
    resp = str(gr_test.function())
    #resp = str("test")
    ack = prod.send('gr2rb_topic', bytes(resp, 'utf-8'))
    print(ack.get())