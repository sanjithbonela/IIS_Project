from kafka import KafkaProducer
from end_to_end import e2e_test

if __name__ == '__main__':
    prod = KafkaProducer(bootstrap_servers=['localhost:9092'])
    resp = str(e2e_test.test_video(path))
    prod.send('my_topic', bytes(resp, 'utf-8'))
    print('done')