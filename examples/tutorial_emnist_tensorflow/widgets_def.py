import ipywidgets as widgets
from ipywidgets import Layout

frac_bits_slider = widgets.IntSlider(
    value=7,
    min=0,
    max=7,
    step=1,
    description='Frac bits:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
    layout=Layout(width='300px')
)

offset_slider = widgets.FloatSlider(
    value=128.0,
    min=0,
    max=128.0,
    step=1.0,
    description='Offset:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
    layout=Layout(width='300px')
)

batch_slider = widgets.IntSlider(
    value=128,
    min=1,
    max=1000,
    step=1,
    description='Batch size:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

epochs_slider = widgets.IntSlider(
    value=1,
    min=1,
    max=20,
    step=1,
    description='Epochs:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

calibration_slider = widgets.IntSlider(
    value=400,
    min=10,
    max=100000,
    step=100,
    description='Calib samples:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
    layout=Layout(width='500px', height='100px')
) 

test_output_slider = widgets.IntSlider(
    value=100,
    min=10,
    max=100000,
    step=10,
    description='Test samples:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
    layout=Layout(width='500px', height='100px')
) 

kernel_type_toggle = widgets.ToggleButtons(
    options=['fx8', 'fx16', 'fx8w16d'],
    description='Kernel type:',
    disabled=False
)

debug_version_checkbox = widgets.Checkbox(
    value=False,
    description='Debug version',
    disabled=False
)

train_button = widgets.Button(
    description="Train model",
    layout=Layout(width='250px', height='50px')
)

train_output = widgets.Output()

train_radio = widgets.RadioButtons(
    options=['Use Synopsys weights', 'Train and load custom weights', 'Load custom weights'],
    value='Use Synopsys weights',
    disabled=False
)


