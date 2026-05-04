import ipywidgets as widgets
from IPython.display import display, clear_output

input_widgets = {}

for col in X.columns:
    if col in label_encoders:
        options = original_categorical_data[col]
        input_widgets[col] = widgets.Dropdown(
            options=options,
            description=col,
            disabled=False,
            layout=widgets.Layout(width='auto') 
        )
    elif X[col].dtype == 'int64' or X[col].dtype == 'float64':
        min_val = X[col].min()
        max_val = X[col].max()
        input_widgets[col] = widgets.FloatSlider(
            value=(min_val + max_val) / 2,
            min=min_val,
            max=max_val,
            step=1.0,
            description=col,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.0f',
            layout=widgets.Layout(width='auto') 
        )
    else:
        input_widgets[col] = widgets.Text(
            description=col,
            disabled=False,
            layout=widgets.Layout(width='auto')
        )

predict_button = widgets.Button(
    description="Predict Sales",
    button_style='success', 
    layout=widgets.Layout(width='auto')
)
output_area = widgets.Output()

def on_predict_button_clicked(b):
    with output_area:
        clear_output()
        input_data = pd.DataFrame(columns=X.columns)
        new_sample = {}
        for col in X.columns:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    new_sample[col] = le.transform([input_widgets[col].value])[0]
                except ValueError:
                    print(f"Warning: '{input_widgets[col].value}' not seen in training data for {col}. Assigning -1.")
                    new_sample[col] = -1
            else:
                new_sample[col] = input_widgets[col].value
        
        input_df = pd.DataFrame([new_sample])
        
        prediction = model.predict(input_df)
        print(f"Predicted Sales Amount: ${prediction[0]:,.2f}")

predict_button.on_click(on_predict_button_clicked)

title_widget = widgets.HTML(
    value="<h2 style='text-align: center; color: #4CAF50;'>Chocolate Sales Prediction</h2>"
)

num_features = len(input_widgets)
num_rows = (num_features + 1) // 2 

grid = widgets.GridspecLayout(num_rows, 2, width='100%', justify_content='center')

widgets_list = list(input_widgets.values())
for i, w in enumerate(widgets_list):
    row = i // 2
    col = i % 2
    grid[row, col] = w

interface_layout = widgets.VBox([
    title_widget,
    grid,
    widgets.HBox([widgets.Label(value='', layout=widgets.Layout(flex='1')), predict_button, widgets.Label(value='', layout=widgets.Layout(flex='1'))]), # Center the button
    output_area
], layout=widgets.Layout(align_items='center')) 

display(interface_layout)
