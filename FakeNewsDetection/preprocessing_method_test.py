import text_preprocessing


def test_method_1():
    s1 = "HeLLo"
    r1 = "hello"
    assert (text_preprocessing.preprocess_text(s1) == r1)


def test_method_2():
    s1 = "Hello, world!"
    r1 = "hello world"
    assert (text_preprocessing.preprocess_text(s1) == r1)


def test_method_3():
    s1 = "HEY 1 hey 2 heyy 3"
    r1 = "hey hey heyy"
    assert (text_preprocessing.preprocess_text(s1) == r1)


def test_method_4():
    s1 = "I think the cat that lies on a chair over there is very cute!"
    r1 = "think cat lies chair cute"
    assert (text_preprocessing.preprocess_text(s1) == r1)


def test_all():
    test_method_1()
    test_method_2()
    test_method_3()
    test_method_4()
