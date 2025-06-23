import plotly.graph_objects as go

def plot_precision_recall(models, precision, recall):
    """
    Generates a grouped bar chart comparing precision and recall for different models.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=precision,
        y=models,
        marker_color='royalblue',
        name='Precision',
        orientation='h',
        text=[f'{val:.1f}%' for val in precision],
        textposition='inside',
        textfont=dict(size=16, weight='bold')
    ))
    fig.add_trace(go.Bar(
        x=recall,
        y=models,
        marker_color='lightgreen',
        name='Recall',
        orientation='h',
        text=[f'{val:.1f}%' for val in recall],
        textposition='inside',
        textfont=dict(size=16, weight='bold')
    ))
    fig.update_layout(
        title={
            'text': "Model Precision and Recall Comparison",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=24, weight='bold')
        },
        xaxis_title={'text': "Percentage", 'font': dict(size=18, weight='bold')},
        yaxis_title={'text': "Models", 'font': dict(size=18, weight='bold')},
        barmode='group',
        yaxis=dict(tickmode='linear', tickfont=dict(size=14, weight='bold')),
        xaxis=dict(tickfont=dict(size=14, weight='bold')),
        template="plotly_white",
        legend=dict(font=dict(size=16, weight='bold'))
    )
    fig.show()

def plot_map(models, values):
    """
    Generates a bar chart showing mAP values for different models.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=values,
        marker_color='red',
        text=values,
        texttemplate='%{text}%',
        textposition='outside',
        textfont=dict(size=16, weight='bold'),
    ))
    fig.update_layout(
        title={
            'text': 'Detection Models Performance (mAP)',
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=24, weight='bold')
        },
        xaxis_title={'text': 'Models', 'font': dict(size=18, weight='bold')},
        yaxis_title={'text': 'mAP Value', 'font': dict(size=18, weight='bold')},
        xaxis=dict(tickfont=dict(size=14, weight='bold')),
        yaxis=dict(
            range=[0, 105],
            gridcolor='lightgray',
            griddash='dash',
            tickfont=dict(size=16, weight='bold')
        ),
        showlegend=False,
        margin=dict(t=100, l=70, r=40, b=80),
        plot_bgcolor='white'
    )
    fig.show() 