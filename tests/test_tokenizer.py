from tab_transformer_pytorch import TabTokenizer
import pandas as pd


def test_tabtokenizer():

    data = pd.DataFrame(
        {
            'col1': pd.Series(['a', 'b', 'c'], dtype='category'),
            'col2': pd.Series(['d', 'e', 'f'], dtype='category')
        }
    )

    test_data = pd.DataFrame(
        {
            'col1': pd.Series(['a', 'b', 'c', 'd', 'a'], dtype='category'),
            'col2': pd.Series(['d', 'e', 'f', None, 'd'], dtype='category')
        }
    )

    expected_result = pd.DataFrame(
        {
            'col1': pd.Series([1, 2, 3, 0, 1], dtype='int8'),
            'col2': pd.Series([5, 6, 7, 4, 5], dtype='int8')
        }
    )

    tokenizer = TabTokenizer()
    tokenizer.fit(data)
    encoded_data = tokenizer.transform(test_data)

    pd.testing.assert_frame_equal(encoded_data, expected_result)
    assert tokenizer.vocab_len == 8