def predict(model, title, tokenizer, pad_sequences, max_length):

    test_x = [title, 'aaaa']

    word_index = tokenizer.word_index

    test_sequences = tokenizer.texts_to_sequences(test_x)
    test_padded_titles = pad_sequences(test_sequences, maxlen=max_length, padding='post')

    #reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    #def decode_review(text):
    #    return ' '.join([reverse_word_index.get(i, '?') for i in text])

    #print(decode_review(padded_titles[200]))

    #loading last  checkpoint
    predictions, pad = model.predict(test_padded_titles)

    return predictions, test_sequences[0], test_padded_titles[0]
