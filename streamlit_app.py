import torch
from src.model import Model
import streamlit as st
import pandas as pd


DATA_DIR = 'data/train'
MODEL_PATH = 'model.pth'


@st.cache_data 
def load_model():
    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH))
    return model
    

@st.cache_data 
def load_data():
    train_text = pd.read_fwf(filepath_or_buffer=f"{DATA_DIR}/text.txt", header=None)
    train_tags = pd.read_fwf(filepath_or_buffer=f"{DATA_DIR}/labels.txt", header=None)[0].map(lambda x: "Harmful" if x == 1 else "Non-harmful")

    train_data = pd.concat([train_text, train_tags], axis=1)
    train_data.columns = ['text', 'tag']

    return train_data


def get_sample_harmful_data(n_rows, train_data):
    return train_data[train_data['tag'] == 'Harmful'].sample(n_rows)['text']

    
def get_sample_non_harmful_data(n_rows, train_data):
    return train_data[train_data['tag'] == 'Non-harmful'].sample(n_rows)['text']


class App:
    def __init__(self) -> None:
        self.model = load_model()
        self.train_data = load_data()

    def run(self):
        st.title("Detekcja mowy nienawiści")
        st.header("Opis aplikacji")
        st.write("Aplikacja służy do wykrywania mowy nienawiści w komentarzach/wiadomociach z czatu. Aby to zrobić, wpisz komentarz w polu tekstowym i naciśnij przycisk 'Sprawdź'.")


        st.header("Opis projektu")
        st.subheader("Cel biznesowy")
        # TODO: pewnie do poprawy xD
        st.write("Celem biznesowym projektu jest stworzenie aplikacji, która będzie w stanie wykrywać mowę nienawiści w komentarzach/wiadomościach z czatu. Aplikacja będzie przydatna dla osób, które chcą uniknąć kontaktu z mową nienawiści w internecie.")

        st.subheader("Cel analityczny")
        st.write("Celem analitycznym jest wykrywanie klasy pozytywnej i osiągnięcie jak największych wartości trafności i metryki G-mean. Progi metryk od których można uznać projekt za udany to: 0.9 dla trafności i 0.8 dla G-mean.")

        st.subheader("Wykorzystane narzędzia")
        st.markdown("* dvc - narzędzie do zarządzania danymi i eksperymentami")
        st.markdown("* nlpaug - biblioteka do augmentacji danych tekstowych")
        st.markdown("* streamlit - biblioteka do tworzenia prostych aplikacji internetowych dotyczących uczenia maszynowego i analizy danych")


        st.subheader("Opis zbioru danych")
        st.write('Wykorzystywany zbiór danych pochodzi z PolEval2019 z zadania 6.1: "Harmful vs non-harmful"(http://2019.poleval.pl/index.php/tasks/task6). Jest to zbiór danych z komentarzami z czatu, które zostały oznaczone jako mowa nienawiści lub nie. Jest on bardzo niezbalansowany, co można zauważyć po następującym wykresie:')
        st.bar_chart(self.train_data['tag'].value_counts())

        st.markdown("**Przykładowe komentarze:**")
        n_rows = st.slider("Liczba komentarzy niezawierających mowy nienawiści", min_value=1, max_value=min(self.train_data[self.train_data['tag']=="Non-harmful"].shape[0], 20), value=5)
        st.table(get_sample_non_harmful_data(n_rows, self.train_data))

        n_rows = st.slider("Liczba komentarzy zawierających mowę nienawiści", min_value=1, max_value=min(self.train_data[self.train_data['tag']=="Harmful"].shape[0], 20), value=5)
        st.table(get_sample_harmful_data(n_rows, self.train_data))

        st.subheader("Wykorzystane podejścia")
        #TODO

        st.header("Demonstracja")
        text = st.text_area("Wprowadź treść")
        if st.button("Sprawdź"):
            embeddings = self.model.tokenizer(text,
                                                padding='max_length',
                                                add_special_tokens=True,
                                                return_tensors="pt"
                                            )
            prediction = self.model(embeddings)
            print(prediction)
            prediction = torch.round(prediction).item()
            prediction_label = "Mowa nienawiści" if prediction == 1 else "Brak mowy nienawiści"
            prediction_border_color = "red" if prediction == 1 else "green"
            # html border element
            col1, col2 = st.columns(2)
            col1.write(f"Komentarz został sklasyfikowany jako:")
            col2.markdown(f"<div style='border: 2px solid {prediction_border_color}; padding: 10px; border-radius: 5px;'>{prediction_label}</div>", unsafe_allow_html=True)



if __name__ == "__main__":
    app = App()
    app.run()