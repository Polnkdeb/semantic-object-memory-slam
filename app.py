from pathlib import Path
import json

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image


RUNS_DIR = Path("runs")


st.set_page_config(
    page_title="Semantic Object Memory SLAM",
    layout="wide"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_scenes():
    if not RUNS_DIR.exists():
        return []

    return sorted([
        p for p in RUNS_DIR.iterdir()
        if p.is_dir()
    ])


def main():
    st.title("Семантическая объектная память по видео")
    st.caption(
        "Демо-проект: визуальная одометрия + детекция объектов + объектная память"
    )

    st.markdown(
        """
        Эта программа анализирует видео, оценивает примерное движение камеры,
        находит объекты на кадрах и сохраняет, где эти объекты были замечены.
        """
    )

    scenes = get_scenes()

    if not scenes:
        st.error("Папка runs/ пуста. Сначала запусти: python src/run_video.py")
        return

    scene_names = [s.name for s in scenes]
    selected_name = st.sidebar.selectbox("Выберите видео/сцену", scene_names)

    scene_dir = RUNS_DIR / selected_name
    frames_dir = scene_dir / "frames"
    outputs_dir = scene_dir / "outputs"

    poses_path = outputs_dir / "poses.json"
    memory_path = outputs_dir / "memory.json"
    objects_path = outputs_dir / "objects.json"

    if not poses_path.exists() or not memory_path.exists():
        st.error(
            "Не найдены poses.json или memory.json. "
            "Сначала запусти полный пайплайн: python src/run_video.py"
        )
        return

    poses = load_json(poses_path)
    memory = load_json(memory_path)

    objects = []
    if objects_path.exists():
        objects = load_json(objects_path)

    pose_df = pd.DataFrame(poses)

    memory_df = pd.DataFrame([
        {
            "Класс объекта": item["label"],
            "Уверенность модели": item["confidence"],
            "Кадр": item["frame"],
            "X камеры": item["camera_position"]["x"],
            "Y камеры": item["camera_position"]["y"],
            "Z камеры": item["camera_position"]["z"],
        }
        for item in memory
    ])

    st.sidebar.markdown("## Фильтры")

    labels = sorted(memory_df["Класс объекта"].unique()) if len(memory_df) else []

    selected_labels = st.sidebar.multiselect(
        "Какие объекты показывать",
        labels,
        default=labels
    )

    min_conf = st.sidebar.slider(
        "Минимальная уверенность детектора",
        0.0,
        1.0,
        0.4,
        0.05
    )

    filtered_df = memory_df[
        (memory_df["Класс объекта"].isin(selected_labels)) &
        (memory_df["Уверенность модели"] >= min_conf)
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric("Кадров обработано", len(poses))
    col2.metric("Наблюдений объектов", len(memory_df))
    col3.metric("Типов объектов", memory_df["Класс объекта"].nunique())

    st.markdown("## Интерактивная семантическая карта")

    st.markdown(
        """
        На карте показана примерная траектория камеры и места, где были замечены объекты.

        **Синяя линия** — движение камеры, восстановленное методом визуальной одометрии.  
        **Точки объектов** — кадры, в которых YOLO обнаружил соответствующие объекты.  
        Координаты не являются точными метрами: монокулярная одометрия восстанавливает траекторию
        с неопределённым масштабом.
        """
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pose_df["x"],
        y=pose_df["z"],
        mode="lines+markers",
        name="Траектория камеры",
        marker=dict(size=5),
        line=dict(width=3),
        text=pose_df["frame"],
        hovertemplate=(
            "Кадр: %{text}<br>"
            "X: %{x:.2f}<br>"
            "Z: %{y:.2f}<extra></extra>"
        ),
    ))

    if len(filtered_df):
        for label in sorted(filtered_df["Класс объекта"].unique()):
            part = filtered_df[filtered_df["Класс объекта"] == label]

            fig.add_trace(go.Scatter(
                x=part["X камеры"],
                y=part["Z камеры"],
                mode="markers",
                name=label,
                marker=dict(size=10),
                text=part["Кадр"],
                customdata=part[["Уверенность модели"]],
                hovertemplate=(
                    f"Объект: {label}<br>"
                    "Кадр: %{text}<br>"
                    "Уверенность: %{customdata[0]:.2f}<br>"
                    "X: %{x:.2f}<br>"
                    "Z: %{y:.2f}<extra></extra>"
                ),
            ))

    fig.update_layout(
        title="Семантическая карта объектов",
        xaxis_title="X координата камеры",
        yaxis_title="Z координата камеры",
        height=650,
        legend_title="Объекты",
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Таблица наблюдений объектов")

    st.markdown(
        """
        Эта таблица показывает **все отдельные обнаружения объектов** на кадрах.

        Каждая строка — это одно наблюдение объекта:
        модель YOLO нашла объект на конкретном кадре и сохранила примерную позицию камеры
        в момент этого наблюдения.

        **Что означают столбцы:**

        - **Класс объекта** — какой объект обнаружен моделью, например `person`, `chair`, `potted plant`.
        - **Уверенность модели** — насколько модель уверена в обнаружении.
        - **Кадр** — изображение, на котором объект был найден.
        - **X/Y/Z камеры** — примерная позиция камеры в момент наблюдения.
        """
    )

    st.dataframe(
        filtered_df.sort_values("Уверенность модели", ascending=False),
        use_container_width=True
    )

    if objects:
        st.markdown("## Таблица устойчивых объектов памяти")

        st.markdown(
            """
            Эта таблица показывает уже не отдельные наблюдения, а **сгруппированные объекты**.

            Например, если один и тот же цветок был найден на нескольких соседних кадрах,
            система объединяет эти наблюдения в один объект памяти.

            **Что означают столбцы:**

            - **ID объекта** — внутренний номер найденного устойчивого объекта.
            - **Класс объекта** — тип объекта.
            - **Количество наблюдений** — сколько раз этот объект был замечен на разных кадрах.
            - **Максимальная уверенность** — самая высокая уверенность YOLO среди всех наблюдений объекта.
            - **X/Y/Z средней позиции** — усреднённая позиция камеры, из которой объект наблюдался.
            """
        )

        objects_df = pd.DataFrame([
            {
                "ID объекта": obj["object_id"],
                "Класс объекта": obj["label"],
                "Количество наблюдений": len(obj["observations"]),
                "Максимальная уверенность": obj["max_confidence"],
                "X средней позиции": obj["mean_position"]["x"],
                "Y средней позиции": obj["mean_position"]["y"],
                "Z средней позиции": obj["mean_position"]["z"],
            }
            for obj in objects
        ])

        st.dataframe(
            objects_df.sort_values("Количество наблюдений", ascending=False),
            use_container_width=True
        )

    st.markdown("## Просмотр кадра")

    st.markdown(
        """
        Здесь можно выбрать кадр, на котором был найден объект,
        и посмотреть исходное изображение из видео.
        """
    )

    if len(filtered_df):
        selected_frame = st.selectbox(
            "Выберите кадр",
            sorted(filtered_df["Кадр"].unique())
        )

        frame_path = frames_dir / selected_frame

        if frame_path.exists():
            image = Image.open(frame_path)
            st.image(
                image,
                caption=f"Кадр {selected_frame}",
                use_container_width=True
            )


if __name__ == "__main__":
    main()