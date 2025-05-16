import pandas as pd
import streamlit as st
import numpy as np
import numpy_financial as npf
import plotly.express as px
import json

st.set_page_config(layout="wide")
st.title("🔆 Solar System Simulation + 25-Year Financial Model")

# --- Upload/Download Input Parameters ---
st.sidebar.title("💾 Save or Load Inputs")

uploaded_params = st.sidebar.file_uploader("📤 Upload Parameters (.json)", type="json")
if uploaded_params:
    uploaded_config = json.load(uploaded_params)
    for k, v in uploaded_config.items():
        st.session_state[k] = v
    st.sidebar.success("Inputs loaded from file!")





# --- Upload Section ---
st.header("1. Upload Load and PV Data")
col1, col2 = st.columns(2)
with col1:
    load_file = st.file_uploader("Upload Load Profile (CSV)", type="csv")
with col2:
    pv_file = st.file_uploader("Upload PV Output Data (CSV)", type="csv")

# --- System Parameters ---
st.header("2. System Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    dc_size = st.number_input("DC System Size (kW)", value=st.session_state.get("dc_size", 40.0))
    base_dc_size = st.number_input("Base DC Size in PV File (kW)", value=st.session_state.get("base_dc_size", 40.0))
with col2:
    inverter_size = st.number_input("Inverter Capacity (kW)", value=st.session_state.get("inverter_size", 30.0))
    inverter_eff = st.number_input("Inverter Efficiency (%)", value=st.session_state.get("inverter_eff", 98.0)) / 100
with col3:
    export_limit = st.number_input("Export Limit (kW)", value=st.session_state.get("export_limit", 30.0))

# --- Utility Rates ---
st.header("3. Utility Tariff Inputs")
col1, col2 = st.columns(2)
with col1:
    import_rate = st.number_input("Import rate (£/kWh)", min_value=0.1, value=st.session_state.get("import_rate", 0.48), step=0.01)
with col2:
    export_rate = st.number_input("Export rate (£/kWh)", min_value=0.01, value=st.session_state.get("export_rate", 0.05), step=0.005)

# --- Financial Parameters ---
st.header("4. Financial Assumptions")
col1, col2, col3 = st.columns(3)
with col1:
    capex_per_kw = st.number_input("Capex (Cost per kW)", value=st.session_state.get("capex_per_kw", 650.0))
    o_and_m_rate = st.number_input("O&M Cost (% of Capex per year)", value=st.session_state.get("o_and_m_rate", 1.0)) / 100
with col2:
    apply_degradation = st.checkbox("Apply Degradation", value=st.session_state.get("apply_degradation", True))
    degradation_rate =  st.number_input("Degradation per Year (%)", value=st.session_state.get("degradation_rate", 0.7)) / 100
with col3:
    import_esc = st.number_input("Import Tariff Escalation (%/year)", value=st.session_state.get("import_esc", 2.0)) / 100
    export_esc = st.number_input("Export Tariff Escalation (%/year)", value=st.session_state.get("export_esc", 1.0)) / 100
    inflation = st.number_input("General Inflation Rate (%/year)", value=st.session_state.get("inflation", 3.0)) / 100
    esc_year = st.number_input("Electricity Inflation from year ", value =st.session_state.get("esc_year",8.0))

# --- Save Current Input Parameters ---
if st.sidebar.button("📥 Save Inputs"):
    input_params = {
        "dc_size": dc_size,
        "base_dc_size": base_dc_size,
        "inverter_size":inverter_size,
        "inverter_eff": inverter_eff,
        "export_limit": export_limit,
        "import_rate": import_rate,
        "export_rate": export_rate,
        "capex_per_kw": capex_per_kw,
        "o_and_m_rate": o_and_m_rate,
        "apply_degradation": apply_degradation,
        "degradation_rate": degradation_rate,
        "import_esc": import_esc,
        "export_esc": export_esc,
        "inflation": inflation,
        "esc_year":esc_year
    }

    json_string = json.dumps(input_params, indent=2)
    st.sidebar.download_button("⬇️ Download JSON", json_string, file_name="saved_inputs.json", mime="application/json")


# --- Run Simulation ---
if load_file and pv_file:
    load_df = pd.read_csv(load_file)
    pv_df = pd.read_csv(pv_file)

    df = pd.DataFrame()
    df['Time'] = pd.to_datetime(load_df.iloc[:, 0], dayfirst=True)
    df['Load'] = load_df.iloc[:, 1]
    df['PV_base'] = pv_df.iloc[:, 1]

    scaling_factor = dc_size / base_dc_size
    df['PV_Prod'] = df['PV_base'] * scaling_factor
    df['Inv_Limit'] = inverter_size
    df['Clipped'] = (df['PV_Prod'] - df['Inv_Limit']).clip(lower=0)
    df['E_Inv'] = df[['PV_Prod', 'Inv_Limit']].min(axis=1)
    df['E_Use'] = df['E_Inv'] * inverter_eff
    df['Inv_Loss'] = df['E_Inv'] * (1 - inverter_eff)
    df['PV_to_Load'] = df[['E_Use', 'Load']].min(axis=1)
    df['Import'] = (df['Load'] - df['PV_to_Load']).clip(lower=0)
    df['Export'] = (df['E_Use'] - df['PV_to_Load']).clip(lower=0).clip(upper=export_limit)
    df['Excess'] = (df['E_Use'] - df['PV_to_Load'] - df['Export']).clip(lower=0)

    total_pv = df['PV_Prod'].sum()
    total_import = df['Import'].sum()
    total_export = df['Export'].sum()
    total_load = df['Load'].sum()
    base_self_use = df['PV_to_Load'].sum()
    base_export = df['Export'].sum()
    base_self_use_ratio = base_self_use / total_pv if total_pv > 0 else 0
    base_export_ratio = base_export / total_pv if total_pv > 0 else 0
    total_clipped = df['Clipped'].sum()
    total_excess = df['Excess'].sum()
    total_inv_loss = df['Inv_Loss'].sum()
    pv_after_losses = total_pv - total_clipped - total_excess - total_inv_loss
    total_solar_clip = total_clipped + total_excess + total_inv_loss
    loss_in_energy = total_solar_clip / total_pv
    direct_consumption = (base_self_use/pv_after_losses)
    specific_production = total_pv/dc_size

    with st.expander("☀️Solar Simulation Results📊", expanded=False):
        row1 = st.columns(4)
        row1[0].metric("Total PV Production (kWh)", f"{total_pv:.2f}")
        row1[1].metric("Grid Import (kWh)", f"{total_import:.2f}")
        row1[2].metric("Exported Energy (kWh)", f"{total_export:.2f}")
        row1[3].metric("Total Load (kWh)", f"{total_load:.2f}")

        row2=st.columns(4)
        row2[0].metric("PV used on site (kWh)",f"{base_self_use:.2f}")
        row2[1].metric("Cipped Energy (kWh)",f"{total_clipped:.2f}")
        row2[2].metric("Excess Energy (kWh)",f"{total_excess:.2f}")
        row2[3].metric("Inverter Losses (kWh)",f"{total_inv_loss:.2f}")

        row3=st.columns(4)
        row3[0].metric("PV used on site (%)",f"{(base_self_use/total_load)*100:.2f}%")
        row3[1].metric("Exported Energy (%)",f"{base_export_ratio * 100:.2f}%")
        row3[2].metric("Imported Energy (%)",f"{(total_import/total_load)*100:.2f}%")
        row3[3].metric("Self Consumption (%)",f"{base_self_use_ratio*100:.2f}%")

        row4=st.columns(4)
        row4[0].metric("Excess Energy (%)",f"{(total_excess/total_pv)*100:.2f}%")
        row4[1].metric("Clipped Energy (%)",f"{(total_clipped/total_pv) * 100:.2f}%")
        row4[2].metric("Inverter Losses (%)",f"{(total_inv_loss/total_pv)*100:.2f}%")
        row4[3].metric("Yield (kWh/kWp)",f"{total_pv/dc_size:.2f}")


        df['Date'] = df['Time'].dt.date
        df['Hour'] = df['Time'].dt.hour

        daily_summary = df.groupby('Date').agg({
            'Load': 'sum',
            'PV_Prod': 'sum',
            'PV_to_Load': 'sum',
            'Import': 'sum',
            'Export': 'sum',
            'Excess': 'sum',
            'Clipped': 'sum',
            'Inv_Loss': 'sum'
        }).reset_index()

        fig1 = px.line(daily_summary, x='Date', y=['Load', 'PV_Prod', 'PV_to_Load'], title="Daily Load vs PV")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(daily_summary, x='Date', y=['Import', 'Export', 'Excess'], title="Import, Export, Excess")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.line(daily_summary, x='Date', y=['PV_Prod', 'Clipped', 'Inv_Loss'], title="Production Losses")
        st.plotly_chart(fig3, use_container_width=True)

        sim_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Solar Simulation CSV", sim_csv, "solar_simulation.csv", "text/csv")

    # --- Financial Projection ---
    st.header("5. 25-Year Financial Results")
    initial_capex = dc_size * capex_per_kw
    years = list(range(26))
    degradation_factors = [(1 - degradation_rate) ** (y - 1) if apply_degradation and y > 0 else 1.0 for y in years]

    cashflow = []
    cumulative = -initial_capex

    for y in years:
        if y == 0:
            cashflow.append({
                "Year": 0,
                "System Price (£)": -initial_capex,
                "O&M Costs (£)": 0,
                "Net Bill Savings (£)": 0,
                "Export Income (£)": 0,
                "Annual Cash Flow (£)": -initial_capex,
                "Cumulative Cash Flow (£)": -initial_capex,
                "PV Production":pv_after_losses,
                "Export Energy":total_export,
                "Import rates":import_rate,
                "Export rates":export_rate
            })
            continue

        deg = degradation_factors[y]
        pv_prod = (dc_size * specific_production - total_solar_clip) * deg
        pv_to_load = pv_prod * direct_consumption
        pv_export = pv_prod - pv_to_load
        import_required = total_load - pv_to_load

        imp_price = import_rate * ((1 + import_esc) ** max(0, y - esc_year))
        exp_price = export_rate * ((1 + export_esc) ** max(0,y - esc_year))

        savings = (total_load - import_required) * imp_price
        export_income = pv_export * exp_price
        om = initial_capex * o_and_m_rate * ((1 + inflation) ** (y - 1))

        annual_cashflow = savings + export_income - om
        cumulative += annual_cashflow

        cashflow.append({
            "Year": y,
            "System Price (£)": -initial_capex if y == 0 else 0,
            "O&M Costs (£)": -om if y > 0 else 0,
            "Net Bill Savings (£)": savings,
            "Export Income (£)": export_income,
            "Annual Cash Flow (£)": annual_cashflow,
            "Cumulative Cash Flow (£)": cumulative,
            "PV Production":pv_prod,
            "Export Energy":pv_export,
            "Import rates":imp_price,
            "Export rates":exp_price
        })

    fin_df = pd.DataFrame(cashflow)
    irr = npf.irr(fin_df['Annual Cash Flow (£)'])
    roi = (fin_df['Cumulative Cash Flow (£)'].iloc[-1] + initial_capex) / initial_capex

    payback = None
    payback_display = "Not achieved"
    for i in range(1, len(fin_df)):
        if fin_df.loc[i, 'Cumulative Cash Flow (£)'] >= 0:
            prev_cum = fin_df.loc[i - 1, 'Cumulative Cash Flow (£)']
            annual_cash = fin_df.loc[i, 'Annual Cash Flow (£)']
            if annual_cash != 0:
                payback = i - 1 + abs(prev_cum) / annual_cash
                years = int(payback)
                months = int(round((payback - years) * 12))
                payback_display = f"{years} years {months} months"
            break

    lcoe = initial_capex / sum([total_pv * d for d in degradation_factors[1:]])

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Initial Capex (£)", f"{initial_capex:,.2f}")
    col2.metric("Payback Period", payback_display)
    col3.metric("ROI (%)", f"{roi * 100:.2f}")
    col4.metric("IRR (%)", f"{irr * 100:.2f}")
    col5.metric("LCOE (£/kWh)", f"{lcoe:.2f}")
  
    with st.expander("📋 Show Cash Flow Table"):
        st.dataframe(fin_df.style.format({
            "System Price (£)": "£{:,.2f}",
            "O&M Costs (£)": "£{:,.2f}",
            "Net Bill Savings (£)": "£{:,.2f}",
            "Export Income (£)": "£{:,.2f}",
            "Annual Cash Flow (£)": "£{:,.2f}",
            "Cumulative Cash Flow (£)": "£{:,.2f}"
        }))

    with st.expander("💰Financial Chart📈"):
     st.plotly_chart(px.bar(fin_df[1:], x='Year', y='Annual Cash Flow (£)', title="Annual Cash Flow"), use_container_width=True)
     st.plotly_chart(px.line(fin_df[1:], x='Year', y='Cumulative Cash Flow (£)', title="Cumulative Cash Flow"), use_container_width=True)

    csv = fin_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cash Flow Table", csv, "cashflow_25yr.csv", "text/csv")

    with st.expander("📊 Batch Simulation (Compare Multiple Systems)", expanded=False):
     num_systems = st.number_input("How many systems to compare?", min_value=2, max_value=10, value=3, step=1)
     dc_num = st.number_input("DC increment ?", min_value = 10, max_value = 100, value = 10, step = 10)
     ac_num = st.number_input("AC increment ?", min_value = 10, max_value = 100, value = 10, step = 10)

    st.markdown("### System Parameters")
    batch_data = {
        "System": [f"System {i+1}" for i in range(num_systems)],
        "DC Size (kW)": [dc_size + dc_num * i for i in range(num_systems)],
        "AC Size (kW)": [inverter_size + ac_num * i for i in range(num_systems)],
        "Export Limit (kW)": [export_limit for _ in range(num_systems)],
    }

    batch_df = pd.DataFrame(batch_data)

    for i in range(num_systems):
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_df.at[i, "DC Size (kW)"] = st.number_input(f"DC Size - System {i+1}", key=f"dc_{i}", value=batch_df.at[i, "DC Size (kW)"])
        with col2:
            batch_df.at[i, "AC Size (kW)"] = st.number_input(f"AC Size - System {i+1}", key=f"ac_{i}", value=batch_df.at[i, "AC Size (kW)"])
        with col3:
            batch_df.at[i, "Export Limit (kW)"] = st.number_input(f"Export Limit - System {i+1}", key=f"exp_{i}", value=batch_df.at[i, "Export Limit (kW)"])

    batch_df["DC/AC Ratio"] = (batch_df["DC Size (kW)"] / batch_df["AC Size (kW)"]).round(2)

    st.dataframe(batch_df)

    # --- Run Batch Calculations ---
    comparison_results = []
    for i, row in batch_df.iterrows():
        dc = row["DC Size (kW)"]
        ac = row["AC Size (kW)"]
        exp_limit = row["Export Limit (kW)"]
        dc_ac_ratio = row["DC/AC Ratio"]
        scaling = dc / base_dc_size

        temp_df = pd.DataFrame()
        temp_df['Load'] = df['Load']
        temp_df['PV_base'] = df['PV_base']
        temp_df['PV_Prod'] = df['PV_base'] * scaling
        temp_df['Inv_Limit'] = ac
        temp_df['E_Inv'] = temp_df[['PV_Prod', 'Inv_Limit']].min(axis=1)
        temp_df['E_Use'] = temp_df['E_Inv'] * inverter_eff
        temp_df['PV_to_Load'] = temp_df[['E_Use', 'Load']].min(axis=1)
        temp_df['Import'] = (temp_df['Load'] - temp_df['PV_to_Load']).clip(lower=0)
        temp_df['Export'] = (temp_df['E_Use'] - temp_df['PV_to_Load']).clip(lower=0).clip(upper=exp_limit)

        total_pv_batch = temp_df['PV_Prod'].sum()
        pv_self = temp_df['PV_to_Load'].sum()
        pv_export = temp_df['Export'].sum()
        self_ratio = (pv_self / total_pv_batch)*100 if total_pv_batch > 0 else 0
        exp_ratio = (pv_export / total_pv_batch)*100 if total_pv_batch > 0 else 0

        capex = dc * capex_per_kw
        om_cost = capex * o_and_m_rate
        net_annual = pv_self * import_rate + pv_export * export_rate - om_cost
        irr = npf.irr([-capex] + [net_annual] * 25)
        roi = ((net_annual * 25) - capex) / capex
        lcoe = capex / total_pv_batch

        cum_cash = -capex
        payback = None
        for yr in range(1, 26):
            cum_cash += net_annual
            if cum_cash >= 0:
                payback = yr
                break

        comparison_results.append({
            "System": f"System {i+1}",
            "DC Size (kW)": dc,
            "AC Size (kW)": ac,
            "Export Limit (kW)": exp_limit,
            "DC/AC Ratio": dc_ac_ratio,
            "Total PV (kWh)": round(total_pv_batch, 1),
            "Self-Use Ratio (%)": round(self_ratio, 2),
            "Export Ratio (%)": round(exp_ratio, 2),
            "Payback (yrs)": payback if payback else "N/A",
            "ROI (%)": round(roi * 100, 1),
            "IRR (%)": round(irr * 100, 1) if irr is not None else "N/A",
            "LCOE (£/kWh)": round(lcoe, 4)
        })

    st.subheader("📋 Batch Comparison Table")
    comp_df = pd.DataFrame(comparison_results)
    st.dataframe(comp_df)

    st.subheader("📈 Compare Metric Across Systems")
    metric_option = st.selectbox("Select Metric to Plot", ["Payback (yrs)", "ROI (%)", "IRR (%)", "LCOE (£/kWh)"])
    fig = px.bar(comp_df, x="System", y=metric_option, title=f"{metric_option} Comparison")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Please upload both Load and PV files to run the simulation.")