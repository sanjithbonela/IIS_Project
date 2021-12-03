from kafka import KafkaConsumer

if __name__ == '__main__':
    consumer = KafkaConsumer('my_topic', group_id='test_grp', bootstrap_servers='localhost:9092', auto_offset_reset = 'earliest')
    for msgs in consumer:
        print(msgs.value.decode('utf-8'))