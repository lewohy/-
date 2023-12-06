# 프로젝트


## 사전 설정

- dh key too small 오류 회피를 위한 python 3.9 버전 설정 필요

### 모듈 설치


```python
%pip install tqdm aiohttp pandas ipywidgets matplotlib plotly openpyxl
```

## 데이터 모델 정의

### Request, Response 모델 정의


```python
class RequestData:
    def __init__(self, title: str, url: str, form_data: dict):
        self.title = title
        self.url = url
        self.form_data = form_data


class ResponseData:
    def __init__(self, title: str, url: str, bytes: bytearray, success: bool):
        self.title = title
        self.url = url
        self.bytes = bytes
        self.success = success
```

## 함수 정의

### async fetch

비동기 요청 함수


```python
import aiohttp
from tqdm.notebook import tqdm

async def fetch(session, request_data: RequestData, position: int) -> ResponseData:
    form_data = aiohttp.FormData(request_data.form_data)

    async with session.post(request_data.url, data=form_data) as response:
        total_size_in_bytes = int(response.headers.get("content-length", 0))

        progress_bar = tqdm(total=total_size_in_bytes, desc=request_data.title, unit="iB", unit_scale=True)

        data = bytearray()
        async for chunk in response.content.iter_any():
            data.extend(chunk)
            progress_bar.update(len(chunk))
        progress_bar.close()

        return ResponseData(request_data.title, request_data.url, data, True)
```

### async fetchAll

비동기 fetch all 함수


```python
import aiohttp
import asyncio
from tqdm.notebook import tqdm


async def fetchAll(requests: list[RequestData]) -> list[ResponseData]:
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index, request_data in enumerate(requests):
            task = asyncio.create_task(fetch(session, request_data, index))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
        return responses
```

### response_data_to_dataframe

ResponseData를 DataFrame으로 변환


```python
import pandas as pd
from io import StringIO
from typing import Callable

def response_data_to_dataframe(responses_data: ResponseData, converter: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
    df = converter(responses_data.bytes)
    df["title"] = responses_data.title

    return df

```

### merge_columns_df

특정 열 기준으로 열 값 합


```python
def merge_columns_df(df: pd.DataFrame, criteria: list[str], column_filter: Callable[[str], bool]) -> pd.DataFrame:
    """
    criteria열과 str열 기준으로 column_filter를 통과하는 열들 합
    """
    filtered_columns = [col for col in df.columns if column_filter(col)]

    return df.groupby(criteria)[filtered_columns].sum().reset_index()
```

## 수집 데이터 정의

### station_request_data

정류소 수집 데이터 정의


```python
station_request_data = RequestData(
    title="서울시버스정류소 위치정보 20231114",
    url="https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false",
    form_data={"infId": "OA-15067", "seqNo": "", "seq": "21", "infSeq": "1"},
)
```

### passengers_request_data_list

승하차 수집 데이터 정의


```python
passengers_request_data_list = [
    RequestData(
        title="2023년 9월",
        url="https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false",
        form_data={"infId": "OA-12913", "seqNo": "", "seq": "75", "infSeq": "3"},
    ),
    RequestData(
        title="2023년 8월",
        url="https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false",
        form_data={"infId": "OA-12913", "seqNo": "", "seq": "74", "infSeq": "3"},
    ),
    RequestData(
        title="2023년 7월",
        url="https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false",
        form_data={"infId": "OA-12913", "seqNo": "", "seq": "73", "infSeq": "3"},
    ),
    RequestData(
        title="2023년 6월",
        url="https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false",
        form_data={"infId": "OA-12913", "seqNo": "", "seq": "72", "infSeq": "3"},
    ),
    RequestData(
        title="2023년 5월",
        url="https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false",
        form_data={"infId": "OA-12913", "seqNo": "", "seq": "71", "infSeq": "3"},
    ),
    RequestData(
        title="2023년 4월",
        url="https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false",
        form_data={"infId": "OA-12913", "seqNo": "", "seq": "70", "infSeq": "3"},
    ),
    RequestData(
        title="2023년 3월",
        url="https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false",
        form_data={"infId": "OA-12913", "seqNo": "", "seq": "69", "infSeq": "3"},
    ),
    RequestData(
        title="2023년 2월",
        url="https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false",
        form_data={"infId": "OA-12913", "seqNo": "", "seq": "68", "infSeq": "3"},
    ),
    RequestData(
        title="2023년 1월",
        url="https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false",
        form_data={"infId": "OA-12913", "seqNo": "", "seq": "67", "infSeq": "3"},
    ),
]
```

## 데이터 수집

### 정류소 데이터 수집


```python
from io import BytesIO

station_df = response_data_to_dataframe(
    (await fetchAll([station_request_data]))[0],
    lambda bytes: (pd.read_excel(BytesIO(bytes), sheet_name="Data", header=0)),
)

station_df.to_csv("station.csv", index=False)
```


```python
print(station_df.head())
```

### 승하차 데이터 수집


```python
passengers_df_list = list(map(
    lambda data: response_data_to_dataframe(data, lambda bytes: pd.read_csv(StringIO(bytes.decode("euc-kr", errors="replace")))),
    await fetchAll(passengers_request_data_list[:])
))

```


```python
for passengers_df in passengers_df_list:
    print(passengers_df.head())
```

## 전처리

### 역별 승하차 수 합

수집한 데이터에서 정류장별 승하차 수 합


```python
merged_passengers_df_list = list(map(
    lambda df: merge_columns_df(
        df, ["사용년월", "표준버스정류장ID"], lambda column: (column.endswith("시승차총승객수") or column.endswith("시하차총승객수"))
    ),
    passengers_df_list,
))
```


```python
for merged_passengers_df in merged_passengers_df_list:
    print(merged_passengers_df.head())
```

### 승하차 데이터 분리

승차 데이터와 하차 데이터 분리


```python
def select_and_rename_columns(df, column_suffix):
    selected_columns = df.loc[:, ["사용년월", "표준버스정류장ID"] + [col for col in df if col.endswith(column_suffix)]]

    new_column_names = ["사용년월", "표준버스정류장ID"] + [
        str(int(col.split("시")[0])) for col in selected_columns.columns if column_suffix in col
    ]
    selected_columns.columns = new_column_names

    return selected_columns
```


```python
board_passengers_df_list = list(map(lambda df: select_and_rename_columns(df, "승차총승객수"), merged_passengers_df_list))
alight_passengers_df_list = list(map(lambda df: select_and_rename_columns(df, "하차총승객수"), merged_passengers_df_list))
```


```python

for board_passengers_df in board_passengers_df_list:
    print(board_passengers_df.head())

for alight_passengers_df in alight_passengers_df_list:
    print(alight_passengers_df.head())
```

### 승하차별 데이터 병합


```python
board_passengers_df = pd.concat(board_passengers_df_list)
alight_passengers_df = pd.concat(alight_passengers_df_list)
```


```python
print(board_passengers_df.head())
print(alight_passengers_df.head())

```

### 승하차 데이터 병합


```python
board_passengers_df['타입'] = "승차"
alight_passengers_df['타입'] = "하차"

passengers_df = pd.concat([board_passengers_df, alight_passengers_df])

print(passengers_df.head())

# save as csv
passengers_df.to_csv("passengers.csv", index=False)

```

### 버스 정류장 데이터와 join


```python
merged_df = pd.merge(station_df, passengers_df, left_on="NODE_ID", right_on="표준버스정류장ID")

print(merged_df.head())
merged_df.to_csv("merged.csv", index=False)
```

## 시각화



```python
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from ipywidgets import widgets

from IPython.display import display

df = merged_df

date_range = sorted(list(df["사용년월"].unique()), reverse=True)

date_slider = widgets.IntSlider(
    description="사용년월: ", min=min(date_range), max=max(date_range), step=1, value=max(date_range)
)
search_textbox = widgets.Text(description="정류소명:   ")
station_dropbox = widgets.Dropdown(description="정류소명:   ", options=df["정류소명"].unique().tolist())

board_trace = go.Bar(name="승차 수", orientation="v", width=0.5)
alight_trace = go.Bar(name="하차 수", orientation="v", width=0.5)

g = go.FigureWidget(
    data=[board_trace, alight_trace],
    layout=go.Layout(
        title=dict(
            text="정류소 시간별 승하차 수",
        ),
        barmode="stack",
    ),
)


def validate():
    if station_dropbox.value in df["정류소명"].unique():
        return True
    else:
        return False


def update_station_dropbox(change):
    search_text = search_textbox.value
    filtered_stations = [station for station in df["정류소명"].unique() if search_text in station]

    with g.batch_update():
        station_dropbox.options = filtered_stations
        if filtered_stations:
            station_dropbox.value = filtered_stations[0]


def response(change):
    if validate():
        boarding_row = df[
            (df["정류소명"] == station_dropbox.value) & (df["사용년월"] == date_slider.value) & (df["타입"] == "승차")
        ].iloc[0]
        alighting_row = df[
            (df["정류소명"] == station_dropbox.value) & (df["사용년월"] == date_slider.value) & (df["타입"] == "하차")
        ].iloc[0]
        board_value = boarding_row.loc[map(str, range(0, 24))]
        alight_value = alighting_row.loc[map(str, range(0, 24))]
        with g.batch_update():
            g.data[0].y = board_value
            g.data[1].y = alight_value
            g.layout.barmode = "stack"
            g.layout.xaxis.title = "시간"
            g.layout.xaxis.tickmode = "linear"
            g.layout.xaxis.dtick = 1
            g.layout.yaxis.title = "승하차 총 승객 수"
            g.layout.yaxis.tickformat = ","


date_slider.observe(response, names="value")
search_textbox.observe(update_station_dropbox, names="value")
station_dropbox.observe(response, names="value")

search_container = widgets.HBox([search_textbox, station_dropbox])

container = widgets.VBox([date_slider, search_container, g])

display(container)

response(None)
```
