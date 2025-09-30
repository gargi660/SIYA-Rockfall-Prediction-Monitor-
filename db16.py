import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SIYA Rockfall Monitor",
                   layout="wide", initial_sidebar_state="expanded")

# Session State
if 'alert_logs' not in st.session_state:
    st.session_state.alert_logs = []
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'data' not in st.session_state:
    st.session_state.data = None

# Parameters
parameters = ["Displacement", "Strain", "Pore Pressure", "Rainfall", 
              "Temperature", "Vibrations", "Tilt", "Humidity", "Wind Speed"]

# ===========================
# Sidebar Navigation
# ===========================
page = st.sidebar.radio("Navigate to:", ["3D Visualization & Map Overview",
                                         "Parameter Graphs", 
                                         "2D Heatmap", 
                                         "Event Logs", 
                                         "Controls"])
# SIYA + Tagline at bottom
st.sidebar.markdown("<br><br><br><h2 style='text-align:center; color:#222; font-weight:bold;'>SIYA</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='text-align:center; color:#555;'>PREDICT. PREVENT. PROTECT.</h4>", unsafe_allow_html=True)

# ===========================
# Sample Data Loader
# ===========================
def load_sample_data():
    now = datetime.now()
    timestamps = pd.date_range(start=now - timedelta(hours=10), end=now, freq="H")
    num_sensors = 5
    sensor_lats = [22.11 + np.random.uniform(-0.02, 0.02) for _ in range(num_sensors)]
    sensor_lons = [82.68 + np.random.uniform(-0.02, 0.02) for _ in range(num_sensors)]
    sensor_ids = [f"S{i+1}" for i in range(num_sensors)]
    
    data_points = len(timestamps)
    displacement_data = np.cumsum(np.random.randn(data_points, num_sensors) * 0.1, axis=0) + np.random.rand(num_sensors) * 5
    strain_data = np.cumsum(np.random.randn(data_points, num_sensors) * 0.05, axis=0) + np.random.rand(num_sensors) * 2
    
    return pd.DataFrame({
        "Timestamp": np.repeat(timestamps, num_sensors),
        "Displacement": displacement_data.flatten(),
        "Strain": strain_data.flatten(),
        "Pore Pressure": np.random.rand(data_points * num_sensors) * 100,
        "Rainfall": np.random.randint(0, 10, data_points * num_sensors),
        "Temperature": np.random.randint(15, 35, data_points * num_sensors),
        "Vibrations": np.random.rand(data_points * num_sensors) * 2,
        "Tilt": np.random.rand(data_points * num_sensors) * 15,
        "Humidity": np.random.randint(30, 95, data_points * num_sensors),
        "Wind Speed": np.random.rand(data_points * num_sensors) * 25,
        "Sensor_ID": np.tile(sensor_ids, data_points),
        "Latitude": np.tile(sensor_lats, data_points),
        "Longitude": np.tile(sensor_lons, data_points),
        "Rockfall_Risk": np.random.uniform(0, 1, data_points * num_sensors),
        "Cause": np.random.choice(["Rainfall", "Displacement", "Vibration", "Strain"], data_points * num_sensors)
    })

def categorize_rockfall_risk(risk_score):
    if risk_score < 0.2:
        return "No Risk", "green"
    elif risk_score < 0.4:
        return "Low Risk", "lightgreen"
    elif risk_score < 0.6:
        return "Moderate", "orange"
    elif risk_score < 0.8:
        return "High", "red"
    else:
        return "Critical", "darkred"

# ===========================
# Data Refresh
# ===========================
if st.session_state.data is None or (datetime.now() - st.session_state.last_refresh).seconds/60 >= 2:
    st.session_state.last_refresh = datetime.now()
    st.session_state.data = load_sample_data()

data = st.session_state.data
data["Risk"], data["Risk_Color"] = zip(*data["Rockfall_Risk"].apply(categorize_rockfall_risk))

# ===========================
# 3D Visualization & Map Overview
# ===========================
if page == "3D Visualization & Map Overview":
    st.header("3D Visualization & Map Overview")
    latest = data.sort_values("Timestamp").groupby("Sensor_ID").last().reset_index()
    
    col_left, col_right = st.columns([1,2])
    
    # Donut Chart
    with col_left:
        st.subheader("Risk Distribution")
        risk_counts = latest["Risk"].value_counts().reindex(
            ["No Risk","Low Risk","Moderate","High","Critical"], fill_value=0)
        fig_donut = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.5,
            marker_colors=["green","lightgreen","orange","red","darkred"],
            texttemplate="%{percent:.1%}",
            textposition="inside",
            showlegend=False
        )])
        fig_donut.update_layout(height=350)
        st.plotly_chart(fig_donut, use_container_width=True)
        
        # Risk Info Below Donut
        risk_prob = latest["Rockfall_Risk"].mean()
        if risk_prob < 0.2: rp_color, rp_text="green","Low"
        elif risk_prob < 0.4: rp_color, rp_text="lightgreen","Moderate Low"
        elif risk_prob < 0.6: rp_color, rp_text="orange","Moderate"
        elif risk_prob < 0.8: rp_color, rp_text="red","High"
        else: rp_color, rp_text="darkred","Critical"
        st.markdown(f"<div style='background-color:{rp_color}; color:white; padding:8px; margin-top:5px;'>Risk Probability: {risk_prob:.1%}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:{rp_color}; color:white; padding:8px;'>Risk Score: {risk_prob:.3f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:{rp_color}; color:white; padding:8px;'>Rockfall Prediction: {rp_text}</div>", unsafe_allow_html=True)
    
    # 3D Terrain
    with col_right:
        st.subheader("3D Terrain with Sensor Locations")
        x = np.linspace(22.09, 22.13, 20)
        y = np.linspace(82.66, 82.70, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(8*(X-22.11)) + np.cos(8*(Y-82.68)) + np.random.normal(0,0.05,X.shape)

        sensor_z = [Z[np.abs(y - row['Longitude']).argmin(), np.abs(x - row['Latitude']).argmin()]+0.01
                    for _, row in latest.iterrows()]

        fig3d = go.Figure(data=[go.Surface(
            z=Z, x=X, y=Y,
            colorscale='Earth',
            opacity=0.7, showscale=False
        )])
        fig3d.add_trace(go.Scatter3d(
            x=latest['Latitude'],
            y=latest['Longitude'],
            z=sensor_z,
            mode='markers',
            marker=dict(
                size=8,
                color=latest['Rockfall_Risk'],
                colorscale='RdYlGn_r',
                cmin=0, cmax=1,
                line=dict(color='black', width=1)
            ),
            text=latest['Sensor_ID'],
            hovertemplate=(
                "Sensor: %{text}<br>"
                "Coordinates: (%{x:.4f}, %{y:.4f})<br>"
                "Risk: %{customdata[0]:.1%}<br>"
                "Cause: %{customdata[1]}<extra></extra>"
            ),
            customdata=np.column_stack((latest['Rockfall_Risk'], latest['Cause']))
        ))

        fig3d.update_layout(
            scene=dict(
                xaxis_title='Latitude',
                yaxis_title='Longitude',
                zaxis_title='Elevation'
            ),
            autosize=True,
            height=600
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # Sensor Map Overview
    st.subheader("Sensor Map Overview")
    map_fig = px.scatter_mapbox(latest, lat="Latitude", lon="Longitude",
                                color="Risk",
                                size=np.clip(latest["Rockfall_Risk"]*20,10,30),
                                hover_name="Sensor_ID",
                                hover_data={"Latitude":True, "Longitude":True, "Risk":True, "Rockfall_Risk":True},
                                color_discrete_map={"No Risk":"green","Low Risk":"lightgreen","Moderate":"orange","High":"red","Critical":"darkred"},
                                zoom=12, height=450)
    map_fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(map_fig, use_container_width=True)

# ===========================
# Parameter Graphs
# ===========================
elif page=="Parameter Graphs":
    st.header("Parameter Graphs")
    choice = st.radio("Select Graph:", ["Single Parameter","All Parameters"])
    
    latest = data.sort_values("Timestamp").groupby("Sensor_ID").last().reset_index()
    
    if choice=="Single Parameter":
        param = st.selectbox("Select Parameter:", parameters)
        st.subheader(f"{param}")
        fig = go.Figure()
        for sensor_id in data["Sensor_ID"].unique():
            sensor_data = data[data["Sensor_ID"]==sensor_id]
            fig.add_trace(go.Scatter(x=sensor_data["Timestamp"], y=sensor_data[param],
                                     mode='lines+markers', name=sensor_id))
        fig.update_layout(height=500, xaxis_title="Time", yaxis_title=param)
        st.plotly_chart(fig, use_container_width=True)
    else:  # All Parameters
        fig_all = go.Figure()
        for param in parameters:
            fig_all.add_trace(go.Bar(x=data["Sensor_ID"].unique(), y=latest[param],
                                     name=param))
        fig_all.update_layout(barmode='group', height=500, xaxis_title="Sensor", yaxis_title="Value")
        st.plotly_chart(fig_all, use_container_width=True)

# ===========================
# 2D Heatmap
# ===========================
elif page=="2D Heatmap":
    st.header("2D Heatmap")
    latest = data.sort_values("Timestamp").groupby("Sensor_ID").last().reset_index()
    x = np.linspace(22.09,22.13,50)
    y = np.linspace(82.66,82.70,50)
    X,Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)
    for i,row in latest.iterrows():
        dist = np.sqrt((X-row['Latitude'])**2 + (Y-row['Longitude'])**2)
        influence = np.exp(-dist*100)
        Z += influence * row['Rockfall_Risk']
    Z = Z/Z.max()
    
    fig2 = go.Figure(data=go.Contour(z=Z, x=x, y=y, colorscale='RdYlGn', contours=dict(coloring='heatmap')))
    for _, row in latest.iterrows():
        fig2.add_trace(go.Scatter(x=[row['Latitude']], y=[row['Longitude']],
                                  mode='markers',
                                  marker=dict(size=8, color=row['Rockfall_Risk'], colorscale='RdYlGn', cmin=0, cmax=1),
                                  text=f"Sensor {row['Sensor_ID']}"))
    fig2.update_layout(title='2D Risk Heatmap', xaxis_title='Latitude', yaxis_title='Longitude', height=500)
    st.plotly_chart(fig2, use_container_width=True)

# ===========================
# Event Logs
# ===========================
elif page=="Event Logs":
    st.header("Event Logs")
    latest = data.sort_values("Timestamp").groupby("Sensor_ID").last().reset_index()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"Time": timestamp, "Risk": latest["Risk"].iloc[0], "Risk Value": f"{latest['Rockfall_Risk'].iloc[0]:.3f}"}
    if not st.session_state.alert_logs or st.session_state.alert_logs[-1] != log_entry:
        st.session_state.alert_logs.append(log_entry)
    logs_df = pd.DataFrame(reversed(st.session_state.alert_logs[-20:]))
    st.dataframe(logs_df, use_container_width=True)

# ===========================
# Controls
# ===========================
elif page=="Controls":
    st.header("Controls")
    if st.button("Acknowledge Alert"):
        st.session_state.alert_logs.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Risk": "Acknowledged",
            "Risk Value": "-"
        })
        st.success("Alert acknowledged!")
    
    # About Project
    if st.checkbox("Show About Project"):
        st.markdown(
            "- Frontend: Streamlit\n"
            "- Location Reference: Kusmunda, Chhattisgarh\n"
            "- ML Model: LightGBM, Random Forest\n"
            "- Meta Learner: Logistic Regression"
        )
