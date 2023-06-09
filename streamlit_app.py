import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.model import Model
from src.utils import preprocessing_text

DATA_DIR = "data/train"
MODEL_PATH = "model.pth"
HATESPEECH_IMAGE = "static/hatespeech.png"
DVC_IMAGE = "static/dvc.png"
NLPAUG_IMAGE = "static/nlp_aug.png"
STREAMLIT_IMAGE = "static/streamlit.png"


@st.cache_data
def load_model():
    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH))
    return model


@st.cache_data
def load_data():
    train_text = pd.read_fwf(filepath_or_buffer=f"{DATA_DIR}/text.txt", header=None)
    train_tags = pd.read_fwf(filepath_or_buffer=f"{DATA_DIR}/labels.txt", header=None)[
        0
    ].map(lambda x: "Harmful" if x == 1 else "Non-harmful")

    train_data = pd.concat([train_text, train_tags], axis=1)
    train_data.columns = ["text", "tag"]

    return train_data


def get_sample_harmful_data(n_rows, train_data):
    return train_data[train_data["tag"] == "Harmful"].sample(n_rows)["text"]


def get_sample_non_harmful_data(n_rows, train_data):
    return train_data[train_data["tag"] == "Non-harmful"].sample(n_rows)["text"]


class App:
    def __init__(self) -> None:
        self.model = load_model()
        self.train_data = load_data()

    def predict(self, text):
        preprocessed_text = preprocessing_text(text)
        embeddings = self.model.tokenizer(
            preprocessed_text,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            max_length=128,
        )
        prediction = self.model(embeddings)
        prediction = torch.round(prediction).item()
        return prediction

    def run(self):
        st.title("Detekcja mowy nienawiści")
        st.header("Opis aplikacji")
        st.write(
            "Aplikacja jest prototypem systemu wykrywania mowy nienawiści w komentarzach/wiadomościach w polskich mediach społecznościowych. W celu jej przetestowania, wpisz komentarz w polu tekstowym i naciśnij przycisk 'Sprawdź'."
        )
        _, col_image, _ = st.columns([1, 8, 1])
        col_image.image(Image.open(HATESPEECH_IMAGE), use_column_width=True)

        st.header("Opis projektu")
        st.subheader("Cel biznesowy")
        st.write("")

        st.write(
            "Celem biznesowym projektu jest stworzenie prototypu systemu, która będzie w stanie wykrywać mowę nienawiści w szeroko rozumianych mediach społecznościowych.",
            "Aplikacja będzie przydatna dla osób, które chcą uniknąć kontaktu z mową nienawiści w internecie i umożliwi filtrowanie lub oznaczanie nieodpowiednich treści.",
            "W ramach naszej pracy skupiliśmy się na jezyku polskim, dla którego nie istnieje tak dużo rozwiązań w przeciwieństwie do języka angielskiego.",
        )

        st.subheader("Cel analityczny")
        st.write(
            "Celem analitycznym jest wykrywanie klasy pozytywnej (mowa nienawiści) i osiągnięcie jak największych wartości trafności i metryki G-mean.",
            "Z racji małej ilości dostępnych danych jest to dość trudne zadanie. Dodatkowo wykorzystywany zbiór z konkursu PolEval2019 jest niezbalansowany, a autorzy podają baseline",
            "na poziomie około 90\% dla trafności. W związku z tym wynik powyżej 90\% trafności oraz 75\% gmean będzie uznany za sukces.",
        )

        st.subheader("Wykorzystane narzędzia")
        col_dvc, col_nlpaug, col_streamlit = st.columns(3)

        col_dvc.image(Image.open(DVC_IMAGE), use_column_width=True)
        col_dvc.markdown("* dvc - narzędzie do zarządzania danymi i eksperymentami")
        col_nlpaug.image(Image.open(NLPAUG_IMAGE), use_column_width=True)
        col_nlpaug.markdown("* nlpaug - biblioteka do augmentacji danych tekstowych")
        col_streamlit.image(Image.open(STREAMLIT_IMAGE), use_column_width=True)
        col_streamlit.markdown(
            "* streamlit - biblioteka do tworzenia prostych aplikacji internetowych dotyczących uczenia maszynowego i analizy danych"
        )

        st.subheader("Opis zbioru danych")
        st.write(
            'Wykorzystywany zbiór danych pochodzi z PolEval2019 z zadania 6.1: "Harmful vs non-harmful"(http://2019.poleval.pl/index.php/tasks/task6). Jest to zbiór danych z komentarzami z czatu, które zostały oznaczone jako mowa nienawiści lub nie. Jest on bardzo niezbalansowany, co można zauważyć po następującym wykresie:'
        )
        st.bar_chart(self.train_data["tag"].value_counts())

        st.markdown("**Przykładowe komentarze:**")
        n_rows = st.slider(
            "Liczba komentarzy niezawierających mowy nienawiści",
            min_value=1,
            max_value=min(
                self.train_data[self.train_data["tag"] == "Non-harmful"].shape[0], 20
            ),
            value=5,
        )
        st.table(get_sample_non_harmful_data(n_rows, self.train_data))

        n_rows = st.slider(
            "Liczba komentarzy zawierających mowę nienawiści",
            min_value=1,
            max_value=min(
                self.train_data[self.train_data["tag"] == "Harmful"].shape[0], 20
            ),
            value=5,
        )
        st.table(get_sample_harmful_data(n_rows, self.train_data))

        st.subheader("Wykorzystane podejścia")
        # TODO

        st.header("Demonstracja")
        text = st.text_area("Wprowadź treść")
        if st.button("Sprawdź"):
            prediction = self.predict(text)
            prediction_label = (
                "Mowa nienawiści" if prediction == 1 else "Brak mowy nienawiści"
            )
            prediction_border_color = "red" if prediction == 1 else "green"
            col1, col2 = st.columns(2)
            col1.write(f"Komentarz został sklasyfikowany jako:")
            col2.markdown(
                f"<div style='border: 2px solid {prediction_border_color}; padding: 10px; border-radius: 5px;'>{prediction_label}</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    app = App()
    app.run()
