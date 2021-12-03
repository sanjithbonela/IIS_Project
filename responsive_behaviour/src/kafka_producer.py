from kafka import KafkaProducer

if __name__ == '__main__':
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
    #producer.send('my_topic', bytes('Hello!', 'utf-8'))
    ack = producer.send(topic='my_topic', value=bytes('Hello!', 'utf-8'))
    print(ack.get())