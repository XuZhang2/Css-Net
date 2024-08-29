from language.vocabulary import AbstractBaseVocabulary
from trainers.abc import AbstractBaseTextEncoder
from models.text_encoders.lstm import SimpleLSTMEncoder, NormalizationLSTMEncoder, SimplerLSTMEncoder
from models.text_encoders.lstm import BertEncoder

TEXT_MODEL_CODES = [SimpleLSTMEncoder.code(), NormalizationLSTMEncoder.code(), SimplerLSTMEncoder.code(),
                    BertEncoder.code()]


def text_encoder_factory(vocabulary: AbstractBaseVocabulary, config: dict) -> AbstractBaseTextEncoder:
    model_code = config['text_encoder']
    feature_size = config['text_feature_size']
    word_embedding_size = config['word_embedding_size']
    lstm_hidden_size = config['lstm_hidden_size']

    if model_code == SimpleLSTMEncoder.code():
        return SimpleLSTMEncoder(vocabulary_len=len(vocabulary), padding_idx=vocabulary.pad_id(),
                                 feature_size=feature_size, word_embedding_size=word_embedding_size,
                                 lstm_hidden_size=lstm_hidden_size)
    elif model_code == NormalizationLSTMEncoder.code():
        return NormalizationLSTMEncoder(vocabulary_len=len(vocabulary), padding_idx=vocabulary.pad_id(),
                                        feature_size=feature_size, norm_scale=config['norm_scale'],
                                        word_embedding_size=word_embedding_size,
                                        lstm_hidden_size=lstm_hidden_size)
    elif model_code == SimplerLSTMEncoder.code():
        return SimplerLSTMEncoder(vocabulary_len=len(vocabulary), padding_idx=vocabulary.pad_id(),
                                  feature_size=feature_size, word_embedding_size=word_embedding_size,
                                  lstm_hidden_size=lstm_hidden_size)
    elif model_code == BertEncoder.code():
        return BertEncoder(feature_size=feature_size, mode=config['mode'])
    else:
        raise ValueError("There's no text encoder matched with {}".format(model_code))
