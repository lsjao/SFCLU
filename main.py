# Standard library imports
import os

# Third-party imports
import folium
from streamlit_folium import folium_static
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image



# Set page config
st.set_page_config(page_title="Official SFCLU", page_icon="📊")

st.title("San Fernando, La Union – Water Supply Project")
st.write("Analyze and visualize the water supply data for San Fernando, La Union. This app provides insights into water demand, willingness to pay (WTP), and other key metrics to support decision-making for water supply projects.")

st.cache_data.clear()


st.divider()
st.write("Upload your CSV file to analyze and visualize the data")
st.set_page_config(layout="wide")


uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load data
    try:
        df = pd.read_csv(uploaded_file)


        st.header("1. General Models and Visualizations")
        st.subheader("1.1 Willingness to Pay (WTP) Analysis")
        st.image('WTP Distribution.png')

        # WTP Summary Statistics
        st.subheader("1.2 Willingness-to-Pay Summary Statistics")
        
        wtp_summary_data = {
            "Metric": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            "WTP_Comm": [316.0000, 155.3481, 45.8642, 20.0000, 160.0000, 160.0000, 160.0000, 600.0000],
            "WTP_Resi": [123.0000, 58.1301, 20.6592, 20.0000, 50.0000, 50.0000, 50.0000, 100.0000]
        }
        wtp_summary_df = pd.DataFrame(wtp_summary_data)
        st.dataframe(wtp_summary_df.style.format({"WTP_Comm": "{:.4f}", "WTP_Resi": "{:.4f}"}))

        # Water Demand Statistics by Classification
        st.subheader("Water Demand Statistics by Classification")
        
        demand_stats_data = {
            "Classification": ["Commercial", "Residential"],
            "Mean": [155.348101, 58.130081],
            "Median": [160.0, 50.0],
            "Standard Deviation": [45.864248, 20.659202],
            "Count": [316, 123]
        }
        demand_stats_df = pd.DataFrame(demand_stats_data)
        st.dataframe(demand_stats_df.style.format({"Mean": "{:.4f}", "Median": "{:.1f}", "Standard Deviation": "{:.4f}", "Count": "{:.0f}"}))


        # Additional WTP Metrics Table
        st.subheader("1.3 Demand vs Cost vs WTP")
        st.image('Violin.png')



        additional_metrics_data = {
            "Metric": ["Water Value (₱)", "Usage Cost (₱)", "WTP (₱)"],
            "Residential": [133.8, 39.9, 58.1],  # Numeric values
            "Commercial": [441.0, 234.0, 155.3]  # Numeric values
        }
        additional_metrics_df = pd.DataFrame(additional_metrics_data)
        st.dataframe(additional_metrics_df.style.format({"Residential": "₱{:.2f}", "Commercial": "₱{:.2f}"}))

        # Average CWSU-4, CWSU-5, and WTP per Barangay and Classification
        st.subheader("1.4 Average Demand, Cost, and WTP per Barangay")

        cwsu_data = {
            "Barangay": [
                "BARANGAY I", "BARANGAY I", "BARANGAY II", "BARANGAY II", "BARANGAY III",
                "BARANGAY III", "BARANGAY IV", "BARANGAY IV", "CARLATAN", "CARLATAN",
                "CATBANGEN", "CATBANGEN", "LINGSAT", "LINGSAT", "MADAYEGDEG",
                "MADAYEGDEG", "PAGDARAOAN", "PAGDARAOAN", "PARIAN", "PARIAN",
                "PORO", "PORO", "SAN AGUSTIN", "SAN AGUSTIN", "SEVILLA",
                "SEVILLA", "TANQUI", "TANQUI"
            ],
            "Classification": [
                "Commercial", "Residential", "Commercial", "Residential", "Commercial",
                "Residential", "Commercial", "Residential", "Commercial", "Residential",
                "Commercial", "Residential", "Commercial", "Residential", "Commercial",
                "Residential", "Commercial", "Residential", "Commercial", "Residential",
                "Commercial", "Residential", "Commercial", "Residential", "Commercial",
                "Residential", "Commercial", "Residential"
            ],
            "Cost": [
                150.00, 72.00, 430.50, 240.00, 282.60,
                91.80, 480.00, 607.50, 1225.20, 39.78,
                570.00, 329.01, 429.00, 120.00, 225.00,
                150.00, 193.50, 316.50, 918.00, 180.00,
                516.00, 120.00, 889.80, 219.00, 1440.00,
                72.00, 549.00, 114.00
            ],
            "Demand": [
                None, 31.50, 1205.91, 36.40, 246.68,
                13.22, 25.05, 38.96, 502.17, 1.36,
                493.18, 146.12, 104.64, 76.93, 28.42,
                38.99, 345.83, 17.78, 83.99, 50.40,
                37.76, 34.25, 20.37, 20.82, 43.67,
                40.43, 24.43, 11.48
            ],
            "WTP": [
                146.67, 80.00, 173.85, 50.00, 156.92,
                57.14, 153.91, 87.50, 110.83, 49.38,
                167.62, 59.00, 159.39, 66.25, 160.00,
                64.44, 166.67, 69.00, 150.83, 31.67,
                156.88, 55.00, 170.00, 50.00, 159.82,
                55.56, 160.00, 50.00
            ]
        }
        awtpdf = pd.DataFrame(cwsu_data)
        
        # Format the DataFrame to show peso signs
        st.dataframe(awtpdf.style.format({
            "Cost": "₱{:,.2f}",
            "Demand": "₱{:,.2f}",
            "WTP": "₱{:,.2f}"
        }))


        # Barangay Latitude and Longitude
        st.subheader("Barangay Latitude (°N) and Longitude (°E)")

        location_data = {
            "Barangay": [
                "BARANGAY I", "BARANGAY II", "BARANGAY III", "BARANGAY IV", "CATBANGEN",
                "CARLATAN", "LINGSAT", "MADAYEGDEG", "PAGDARAOAN", "PARIAN",
                "PORO", "SAN AGUSTIN", "SEVILLA", "TANQUI"
            ],
            "Latitude (°N)": [
                16.618048, 16.615200, 16.618852, 16.616133, 16.609867,
                16.633773, 16.643919, 16.601373, 16.625035, 16.593661,
                16.606856, 16.605968, 16.596318, 16.616012
            ],
            "Longitude (°E)": [
                120.318529, 120.317284, 120.316357, 120.315254, 120.310379,
                120.315452, 120.310484, 120.312349, 120.317222, 120.313593,
                120.304601, 120.300182, 120.321195, 120.323383
            ]
        }
        location_df = pd.DataFrame(location_data)
        st.dataframe(location_df)


        # Ensure barangay names are consistent for merging
        awtpdf['Barangay'] = awtpdf['Barangay'].str.upper()
        location_df['Barangay'] = location_df['Barangay'].str.upper()
        merged_df = pd.merge(awtpdf, location_df, on='Barangay', how='inner')
        combined_df = merged_df.groupby(['Barangay', 'Latitude (°N)', 'Longitude (°E)'], as_index=False).agg({'WTP': 'mean'})
        center_lat, center_lon = combined_df['Latitude (°N)'].mean(), combined_df['Longitude (°E)'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        vmin, vmax = combined_df['WTP'].min(), combined_df['WTP'].max()
        cmap = plt.colormaps['autumn']


        def get_color(wtp):
            norm_wtp = (wtp - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            return mcolors.to_hex(cmap(norm_wtp))

        def get_radius(wtp):
            return 40 * (wtp - vmin + 20) / (vmax - vmin)

        for _, row in combined_df.iterrows():
            folium.CircleMarker(
                location=[row['Latitude (°N)'], row['Longitude (°E)']],
                radius=get_radius(row['WTP']),
                color=get_color(row['WTP']),
                fill=True,
                fill_color=get_color(row['WTP']),
                fill_opacity=0.3,  # Increased opacity for better visibility
                tooltip=f"Barangay: {row['Barangay']}<br>Avg WTP: ₱{row['WTP']:.2f}"
            ).add_to(m)


        st.subheader("Willingness to Pay (WTP) Heatmap")
        st.folium_static(m, width=1000, height=650)
        st.markdown('<div style="height: 1px;"></div>', unsafe_allow_html=True)
        st.write(f"WTP Range: ₱{vmin:.2f} to ₱{vmax:.2f}")

    
        st.divider()

        st.subheader("2. Preliminary Expolatory Data Analysis (EDA)")
        html_file_path = "full-eda-data-profiling.html" 
        if os.path.exists(html_file_path):
            # Read the HTML file
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            # Display the HTML content
            st.components.v1.html(html_content, height=800)  # Adjust height as needed
        else:
            st.error("HTML report not found. Please check the file path.")


        st.divider()
        # Ensure required columns exist
        required_cols = ['Classification', 'Barangay']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Required column '{col}' not found in the CSV file")
                st.stop()
        
        # Classification selection with Residential/Commercial/All options
        st.subheader("Filter Options")
        classification_options = ["Residential", "Commercial"]
        selected_classifications = st.multiselect(
            "Select Classification:", 
            options=classification_options,
            default=classification_options
        )
        
        # Barangay selection from actual data
        barangay_options = sorted(df['Barangay'].unique().tolist())
        barangay_options.insert(0, "All")  # Add "All" option at beginning
        
        selected_barangays = st.multiselect(
            "Select Barangay:",
            options=barangay_options,
            default=["All"]
        )
        
        # Apply filters
        filtered_df = df.copy()
        if selected_classifications and "All" not in selected_classifications:
            filtered_df = filtered_df[filtered_df['Classification'].isin(selected_classifications)]
        if selected_barangays and "All" not in selected_barangays:
            filtered_df = filtered_df[filtered_df['Barangay'].isin(selected_barangays)]
    











        st.dataframe(filtered_df)




        st.header("3. Itemized Visualizations (CWSU & AWTP)")
        # CWSU-1 Bar Graph
        st.subheader("CWSU-1 Primary Usage of Water")
        cwsu1_columns = [
            'CWSU-1 - Usage - Drinking',
            'CWSU-1 - Usage - Food preparation',
            'CWSU-1 - Usage - Cleaning & Sanitation',
            'CWSU-1 - Usage - Manufacturing/Production',
            'CWSU-1 - Usage - Landscaping/Irrigation'
        ]

        # Simplify column names for display
        display_names = {
            'CWSU-1 - Usage - Drinking': 'Drinking',
            'CWSU-1 - Usage - Food preparation': 'Food Prep',
            'CWSU-1 - Usage - Cleaning & Sanitation': 'Cleaning',
            'CWSU-1 - Usage - Manufacturing/Production': 'Manufacturing',
            'CWSU-1 - Usage - Landscaping/Irrigation': 'Landscaping'
        }

        # Check which CWSU columns actually exist in the data
        available_cwsu1_cols = [col for col in cwsu1_columns if col in filtered_df.columns]
        
        if available_cwsu1_cols:
            # Prepare data for visualization
            result_df = pd.DataFrame()
            
            for classification in ['Residential', 'Commercial']:
                if classification in filtered_df['Classification'].unique():
                    # Filter by classification
                    class_df = filtered_df[filtered_df['Classification'] == classification]
                    
                    # Calculate percentages for available columns
                    class_results = {}
                    for col in available_cwsu1_cols:
                        total_count = len(class_df)
                        if total_count > 0:
                            # Count non-null, non-zero responses (assuming 1=Yes, 0=No)
                            yes_count = class_df[col].fillna(0).astype(bool).sum()
                            class_results[display_names[col]] = (yes_count / total_count) * 100
                    
                    # Add to results
                    temp_df = pd.DataFrame.from_dict(class_results, orient='index', columns=[classification])
                    result_df = pd.concat([result_df, temp_df], axis=1)
            
            # Plot the results for CWSU-1
            if not result_df.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Custom colors
                colors = []
                if 'Residential' in result_df.columns:
                    colors.append('#b3e9c7')  # Green
                if 'Commercial' in result_df.columns:
                    colors.append('#8367C7')  # Purple
                
                result_df.plot(kind='bar', color=colors, ax=ax, width=0.8)
                
                # Formatting
                ax.set_title('Usage Percentage by Classification')
                ax.set_ylabel('Percentage (%)')
                ax.set_xlabel('Usage Type')
                ax.set_ylim(0, 100)
                
                # Add percentages on bars
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.1f}%', 
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', 
                                xytext=(0, 5), 
                                textcoords='offset points',
                                fontsize=10)
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No data available for selected filters")
        else:
            st.warning("No CWSU-1 usage data found in the uploaded file")

        # CWSU-2 Bar Graph
        st.subheader("CWSU-2 Presence of Water Treatment by Barangay")

        # Check if there is CWSU-2 Treatment data in the DataFrame
        if 'CWSU-2 - Treatment' in filtered_df.columns:
            # Calculate averages by barangay and classification
            filtered_df['Percentage'] = filtered_df['CWSU-2 - Treatment'] * 100
            
            # Filter columns dynamically based on selected classifications only
            selected_cols = [col for col in ['Residential', 'Commercial'] 
                            if col in selected_classifications or 'All' in selected_classifications]
            
            if selected_cols:  # Only proceed if at least one classification is selected
                result_cwsu2 = pd.pivot_table(filtered_df, 
                                            index='Barangay', 
                                            columns='Classification',
                                            values='Percentage', 
                                            aggfunc='mean')[selected_cols].fillna(0)

                # Calculate overall averages for selected classifications
                overall_avg_cwsu2 = filtered_df.groupby('Classification')['CWSU-2 - Treatment'].mean() * 100
                overall_avg_cwsu2 = overall_avg_cwsu2[selected_cols]

                # Plotting
                fig, ax = plt.subplots(figsize=(14, 7))
                
                # Dynamically set colors for selected classifications
                colors = []
                legend_labels = []
                if 'Residential' in selected_cols:
                    colors.append('#b3e9c7')
                    legend_labels.append('Residential')
                if 'Commercial' in selected_cols:
                    colors.append('#8367c7')
                    legend_labels.append('Commercial')

                result_cwsu2.plot(kind='bar', ax=ax, width=1,
                                color=colors, edgecolor='white')
                
                # Formatting
                plt.title('CWSU-2 Treatment Percentage by Barangay', pad=20)
                plt.ylabel('Average Percentage (%)')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Classification', labels=legend_labels)
                plt.grid(axis='y', linestyle='--', alpha=0.4)

                # Add value labels on bars
                for p in ax.patches:
                    width, height = p.get_width(), p.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.1f}%',
                                xy=(p.get_x() + width/2, height),
                                xytext=(0, 5),
                                textcoords="offset points",
                                ha='center', va='bottom', 
                                fontsize=8, color='black')

                # Add mean lines for selected classifications only
                for avg, label in zip(overall_avg_cwsu2, selected_cols):
                    plt.axhline(y=avg, color='#C79767', linestyle='--', linewidth=1)
                    plt.text(x=len(result_cwsu2.index) + 0.2, y=avg, 
                            s=f'{label} Avg: {avg:.1f}%',
                            color='#C79767', va='center', 
                            ha='left', fontsize=12, fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No classification selected for CWSU-2")
        else:
            st.warning("No CWSU-2 treatment data found in the uploaded file")

        # CWSU-3 Bar Graph - Current Water Sources by Classification
        st.subheader("CWSU-3 Current Water Sources")

        # Define CWSU-3 columns (adjust these to match your actual column names)
        cwsu3_columns = {
            'CWSU-3 - Current Sources - Tap Water (Water District etc.)': 'Tap Water',
            'CWSU-3 - Current Sources - Deep Well (owned)': 'Deep Well',
            'CWSU-3 - Current Sources - Truck Delivery Services (5 m3)': 'Truck Delivery',
            'CWSU-3 - Current Sources - Bottled water (5 gallons)': 'Bottled Water',
            'CWSU-3 - Current Sources - Others': 'Others'
        }

        # Check which CWSU-3 columns actually exist in the data
        available_cwsu3_cols = [col for col in cwsu3_columns if col in filtered_df.columns]

        if available_cwsu3_cols:
            # Prepare data for visualization
            cwsu3_df = filtered_df[available_cwsu3_cols + ['Classification']].copy()
            
            # Count affirmative responses (assuming 1=Yes, 0=No)
            sum_df = cwsu3_df.groupby('Classification').sum().T
            
            # Normalize by sample size (adjust these numbers based on your actual sample counts)
            classification_counts = filtered_df['Classification'].value_counts()
            
            if 'Residential' in sum_df.columns:
                sum_df['Residential'] = (sum_df['Residential'] / classification_counts['Residential']) * 100
            if 'Commercial' in sum_df.columns:
                sum_df['Commercial'] = (sum_df['Commercial'] / classification_counts['Commercial']) * 100
            
            # Plot the results for CWSU-3
            if not sum_df.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Custom colors
                colors = []
                if 'Residential' in sum_df.columns:
                    colors.append('#b3e9c7')  # Green
                if 'Commercial' in sum_df.columns:
                    colors.append('#8367C7')  # Purple
                
                sum_df.plot(kind='bar', color=colors, ax=ax, width=0.8)
                
                # Formatting
                ax.set_title('Current Water Sources by Classification')
                ax.set_ylabel('Affirmative Responses (%)')
                ax.set_xlabel('Water Source')
                ax.set_ylim(0, 100)
                
                # Add percentages on bars
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.1f}%', 
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', 
                                xytext=(0, 5), 
                                textcoords='offset points',
                                fontsize=10)
                ax.set_xticklabels([cwsu3_columns[label] for label in sum_df.index], rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No data available for selected filters in CWSU-3")
        else:
            st.warning("No CWSU-3 water source data found in the uploaded file")





        # CWSU-4 Pie Charts (Residential vs Commercial)
        st.subheader("CWSU-4 Water Source Distribution")

        # Define CWSU-4 data columns - update these to match your actual column names
        cwsu4_cols = [
            'CWSU-4 - Primary Drinking Source - Tap Water (Water District etc.)',
            'CWSU-4 - Primary Drinking Source - Deep Well (owned)',
            'CWSU-4 - Primary Drinking Source - Truck Delivery Services (5 m3)',
            'CWSU-4 - Primary Drinking Source - Bottled water (5 gallons)',
            'CWSU-4 - Primary Drinking Source - Others'
        ]

        # Simplified labels
        simplified_labels = ['Tap Water', 'Deep Well', 'Truck Delivery', 'Bottled Water', 'Others']

        colors_comm = ['#7E63BE', '#8367c7', '#68519C', '#51407A', '#3B2E58']
        colors_res = ['#ACE0BF', '#92BEA2', '#B3E9C7', '#789C85', '#5E7A68']

        # Create a single figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Residential Pie Chart
        if 'Residential' in filtered_df['Classification'].values:
            res_data = filtered_df[filtered_df['Classification'] == 'Residential'][cwsu4_cols].sum()
            ax1.pie(res_data, labels=simplified_labels, autopct='%1.1f%%', colors=colors_res)
            ax1.set_title('Residential Water Sources')

        # Commercial Pie Chart
        if 'Commercial' in filtered_df['Classification'].values:
            com_data = filtered_df[filtered_df['Classification'] == 'Commercial'][cwsu4_cols].sum()
            ax2.pie(com_data, labels=simplified_labels, autopct='%1.1f%%', colors=colors_comm)
            ax2.set_title('Commercial Water Sources')

        # Adjust layout and display
        plt.tight_layout()
        st.pyplot(fig)

        # CWSU-4 Boxplots (Original, Inliers, Outliers)
        st.subheader("CWSU-4 Boxplots")

        # Extract the relevant column and drop NaN values
        col = filtered_df['CWSU-4 - Primary Drinking Source - gals'].dropna()

        # Calculate Q1, Q3, and the bounds for outliers
        q1, q3 = col.quantile([.25, .75])
        b = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)

        # Identify inliers and outliers
        inliers = col[col.between(*b)]
        outliers = col[~col.between(*b)]

        # Create a DataFrame for plotting
        data = pd.DataFrame({
            'Values': pd.concat([col, inliers, outliers]),
            'Category': ['Original'] * len(col) + ['Inliers'] * len(inliers) + ['Outliers'] * len(outliers)
        })

        # Create a figure for the boxplots
        plt.figure(figsize=(12, 6))

        # Create the boxplot
        sns.boxplot(x='Category', y='Values', data=data, palette=["#C79767", "#C2F8CB", "#8367C7"])

        # Formatting
        plt.title('CWSU-4 Boxplots', fontsize=14)
        plt.ylabel('Water Usage (Gallons)')
        plt.xlabel('Category')
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        # Adjust layout and display the plots in Streamlit
        plt.tight_layout()
        st.pyplot(plt)

        # CWSU-4 Bar Graph - Average Water Usage (Gallons)
        st.subheader("CWSU-4 Average Water Usage by Barangay")

        # Check if data exists
        if 'CWSU-4 - Primary Drinking Source - gals' in filtered_df.columns:
            # Calculate averages (keep in gallons, don't convert to %)
            filtered_df['Usage_Gallons'] = filtered_df['CWSU-4 - Primary Drinking Source - gals']
            
            # Filter by selected classifications
            selected_cols = [col for col in ['Residential', 'Commercial'] 
                            if col in selected_classifications or 'All' in selected_classifications]
            
            if selected_cols:
                # Create pivot table with gallons
                result_cwsu4 = pd.pivot_table(filtered_df,
                                            index='Barangay',
                                            columns='Classification',
                                            values='Usage_Gallons',
                                            aggfunc='mean').fillna(0)[selected_cols]

                # Calculate overall averages
                overall_avg = filtered_df.groupby('Classification')['Usage_Gallons'].mean()[selected_cols]

                # Create plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                # Set colors and plot for average usage
                colors = []
                if 'Residential' in selected_cols:
                    colors.append('#b3e9c7')
                if 'Commercial' in selected_cols:
                    colors.append('#8367c7')
                
                result_cwsu4.plot(kind='bar', ax=ax1, color=colors, width=0.8)

                # Add average lines for each classification
                for i, (classification, avg) in enumerate(overall_avg.items()):
                    ax1.axhline(y=avg, 
                                color=colors[i], 
                                linestyle=':', 
                                linewidth=2,
                                label=f'{classification} Avg: {avg:.1f} gal')

                # Formatting for average usage plot
                ax1.set_title('Average Water Usage by Barangay (Gallons)')
                ax1.set_ylabel('Water Usage (Gallons)')
                ax1.legend(loc='upper right')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels for average usage
                for p in ax1.patches:
                    if p.get_height() > 0:
                        ax1.annotate(f'{p.get_height():.1f}', 
                                    (p.get_x() + p.get_width()/2., p.get_height()),
                                    ha='center', va='center', 
                                    xytext=(0, 5),
                                    textcoords='offset points')


                # Calculate log-transformed values (keep in gallons)
                filtered_df['Log_Usage'] = np.log1p(filtered_df['Usage_Gallons'])  # log1p handles log(1+x)

                # Create pivot table with log-transformed gallons
                log_result_cwsu4 = pd.pivot_table(filtered_df,
                                                    index='Barangay',
                                                    columns='Classification',
                                                    values='Log_Usage',
                                                    aggfunc='mean').fillna(0)[selected_cols]

                # Calculate overall averages for log-transformed values
                log_overall_avg = filtered_df.groupby('Classification')['Log_Usage'].mean()[selected_cols]

                # Set colors and plot for log-transformed usage
                log_result_cwsu4.plot(kind='bar', ax=ax2, color=colors, width=0.8)

                # Add average lines for each classification in log-transformed plot
                for i, (classification, avg) in enumerate(log_overall_avg.items()):
                    ax2.axhline(y=avg, 
                                color=colors[i], 
                                linestyle=':', 
                                linewidth=2,
                                label=f'{classification} Avg: {np.expm1(avg):.1f} gal (log scale)')

                # Formatting for log-transformed usage plot
                ax2.set_title('Log-Transformed Average Water Usage by Barangay (Gallons)')
                ax2.set_ylabel('Log-Transformed Water Usage (log(gallons))')
                ax2.legend(loc='upper right')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels for log-transformed usage
                for p in ax2.patches:
                    if p.get_height() > 0:
                        ax2.annotate(f'{np.expm1(p.get_height()):.1f}',  # Convert back to original scale for display
                                    (p.get_x() + p.get_width()/2., p.get_height()),
                                    ha='center', va='center', 
                                    xytext=(0, 5),
                                    textcoords='offset points')

                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.warning("No classification selected")
        else:
            st.warning("CWSU-4 usage data not found in uploaded file")


        # CWSU-5 Visualizations
        st.subheader("CWSU-5 Average Monthly Water")

        if 'CWSU-5 - Ave Demand (in cbm)' in filtered_df.columns:
            filtered_df['CWSU-5 - Ave Demand (in cbm)'] = pd.to_numeric(filtered_df['CWSU-5 - Ave Demand (in cbm)'], errors='coerce')
            
            
            if len(selected_classifications) > 0:
                colors = ['#b3e9c7' if 'Residential' in selected_classifications else '#8367c7',
                        '#8367c7' if 'Commercial' in selected_classifications else '#b3e9c7']
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Standard bar plot
                result = filtered_df.pivot_table(index='Barangay', 
                                                columns='Classification',
                                                values='CWSU-5 - Ave Demand (in cbm)',
                                                aggfunc='mean').fillna(0)
                
                result.plot(kind='bar', ax=ax1, color=colors, width=0.8)
                
                # Add value labels for standard bar plot
                for p in ax1.patches:
                    if p.get_height() > 0:
                        ax1.text(p.get_x() + p.get_width() / 2., p.get_height(),
                                f'{p.get_height():.1f}', ha='center', va='bottom')

                # Add average lines for standard plot
                avg = filtered_df.groupby('Classification')['CWSU-5 - Ave Demand (in cbm)'].mean()
                for i, (cls, val) in enumerate(avg.items()):
                    ax1.axhline(y=val, color=colors[i], linestyle=':', linewidth=2, 
                                label=f'{cls} Avg: {val:.1f} cbm')

                ax1.set_title('Average Water Demand (cbm)')
                ax1.set_ylabel('Demand (cbm)')
                ax1.legend(title='Classification')
                plt.sca(ax1)
                plt.xticks(rotation=45, ha='right')
                
                # Log-Transformed Bar Graph
                filtered_df['Log_Demand'] = np.log1p(filtered_df['CWSU-5 - Ave Demand (in cbm)'])
                log_result = filtered_df.pivot_table(index='Barangay',
                                                    columns='Classification',
                                                    values='Log_Demand',
                                                    aggfunc='mean').fillna(0)
                
                # Create the log-transformed bar plot
                log_bars = log_result.plot(kind='bar', ax=ax2, color=colors, width=0.8)
                
                # Add value labels for log-transformed bar plot
                for p in log_bars.patches:  # Use log_bars to access the patches
                    if p.get_height() > 0:
                        ax2.text(p.get_x() + p.get_width() / 2., p.get_height(),
                                f'{np.expm1(p.get_height()):.1f}', ha='center', va='bottom')

                # Add average lines for log-transformed plot
                log_avg = filtered_df.groupby('Classification')['Log_Demand'].mean()
                for i, (cls, val) in enumerate(log_avg.items()):
                    ax2.axhline(y=val, color=colors[i], linestyle=':', linewidth=2, 
                                label=f'{cls} Avg: {np.expm1(val):.1f} cbm')

                ax2.set_title('Log-Transformed Water Demand')
                ax2.set_ylabel('log(cbm + 1)')
                ax2.legend(title='Classification')
                plt.sca(ax2)
                plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.warning("Please select at least one classification")
        else:
            st.warning("CWSU-5 demand data not found in the dataset")




        # CWSU-6 Box Plots
        st.subheader("CWSU-6 Actual Water Use by Source")

        # Define column names and simplified labels
        columns = [
            'CWSU-6 - Actual Use (m3 / gal) - Tap Water (Water District etc.)',
            'CWSU-6 - Actual Use (m3 / gal) - Deep Well (owned)',
            'CWSU-6 - Actual Use (m3 / gal) - Truck Delivery Services (5 m3)',
            'CWSU-6 - Actual Use (m3 / gal) - Bottled water (5 gallons)',
            'CWSU-6 - Actual Use (m3 / gal) - Others (gallons)'
        ]
        labels = ['Tap Water', 'Deep Well', 'Truck Delivery', 'Bottled Water', 'Others']

        if all(col in filtered_df.columns for col in columns):
            if len(selected_classifications) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Residential Boxplot
                if 'Residential' in selected_classifications:
                    res_data = filtered_df[filtered_df['Classification'] == 'Residential'][columns]
                    bp1 = ax1.boxplot([res_data[col].dropna() for col in columns],
                                    patch_artist=True,
                                    labels=labels)
                    
                    # Set color for Residential boxes
                    for box in bp1['boxes']:
                        box.set(facecolor='#b3e9c7')
                    ax1.set_title('Residential Water Use')
                    ax1.set_ylabel('Usage (m³/gal)')
                
                # Commercial Boxplot
                if 'Commercial' in selected_classifications:
                    com_data = filtered_df[filtered_df['Classification'] == 'Commercial'][columns]
                    bp2 = ax2.boxplot([com_data[col].dropna() for col in columns],
                                    patch_artist=True,
                                    labels=labels)
                    
                    # Set color for Commercial boxes
                    for box in bp2['boxes']:
                        box.set(facecolor='#8367c7')
                    ax2.set_title('Commercial Water Use')
                    ax2.set_ylabel('Usage (m³/gal)')

                plt.xticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Please select at least one classification")
        else:
            st.warning("CWSU-6 usage data not found in dataset")

        # CWSU-7 Peak Months
        st.subheader("CWSU-7 Peak Months")

        # Month mapping
        month_map = {month: i for i, month in enumerate(
            ['January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'], 1)}

        # Create new DataFrame for mapped months
        mapped_months_df = filtered_df[['CWSU-7 - Peak Month - Start', 'CWSU-7 - Peak Month - End']].copy()
        mapped_months_df['CWSU-7 - Peak Month - Start'] = mapped_months_df['CWSU-7 - Peak Month - Start'].map(month_map)
        mapped_months_df['CWSU-7 - Peak Month - End'] = mapped_months_df['CWSU-7 - Peak Month - End'].map(month_map)

        # Plot Peak Water Consumption Periods (Gantt Chart)
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        # Plot each row from the new DataFrame
        for i, (start, end) in enumerate(zip(mapped_months_df['CWSU-7 - Peak Month - Start'], mapped_months_df['CWSU-7 - Peak Month - End'])):
            if pd.notnull(start) and pd.notnull(end):
                ax.plot([start, end], [i, i], color='#8367c7', linewidth=2.5)

        # Styling for Peak Water Consumption Periods
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_map.keys(), rotation=45, ha='right', fontsize=10)
        ax.set_yticks([])
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        plt.title("Peak Water Consumption Periods (Start–End Months)", fontsize=14, weight='bold')
        plt.xlabel("Month", fontsize=12)
        plt.tight_layout()

        # Display Gantt Chart in Streamlit
        st.pyplot(plt)

        # Calculate frequency and percentage for each month
        month_freq = mapped_months_df.apply(pd.Series.value_counts).sum(axis=1).fillna(0)
        percent_freq = (month_freq / month_freq.sum() * 100).round(1).reindex(range(1, 13), fill_value=0)

        # Plot Peak Month Density
        fig, ax = plt.subplots(figsize=(12, 2))
        for i in range(12):
            ax.axvspan(i, i + 1, color='#8367c7', alpha=percent_freq[i + 1] / 100 * 0.8 + 0.2)  # Dynamic opacity
            if percent_freq[i + 1] > 0:
                ax.text(i + 0.5, 0.5, f"{percent_freq[i + 1]}%",
                        ha='center', va='center', color='white', fontweight='bold')

        # Styling for Peak Month Density
        ax.set_xticks(np.arange(0.5, 12.5, 1))
        ax.set_xticklabels(month_map.keys(), rotation=45, ha='right')
        ax.set_yticks([])
        ax.set_xlim(0, 12)
        ax.set_title('Peak Month Distribution (%)', fontsize=14, pad=20, weight='bold')
        ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)

        plt.tight_layout()

        # Display Density Graph in Streamlit
        st.pyplot(fig)



        st.subheader("CWSU-8 Monthly Cost Distribution (%)")

        # Define a mapping for shorter x-axis labels
        cost_mapping = {
            'CWSU-8 - Monthly Costs - Less than PhP 1,000': 'Less than PhP 1K',
            'CWSU-8 - Monthly Costs - PhP 1,000 - PhP 3,000': 'PhP 1K - 3K',
            'CWSU-8 - Monthly Costs - PhP 3,000 - PhP 5,000': 'PhP 3K - 5K',
            'CWSU-8 - Monthly Costs - PhP 5,000 - PhP 10,000': 'PhP 5K - 10K',
            'CWSU-8 - Monthly Costs - PhP 10,000 and above': 'PhP 10K+'
        }

        # Check which CWSU-8 cost columns actually exist in the data
        available_cost_cols = [col for col in cost_mapping if col in filtered_df.columns]

        if available_cost_cols:
            # Prepare data for visualization
            cwsu8_df = filtered_df[available_cost_cols + ['Classification']].copy()
            
            # Count affirmative responses (assuming 1=Yes, 0=No)
            sum_df = cwsu8_df.groupby('Classification').sum().T
            
            # Normalize by sample size (adjust these numbers based on your actual sample counts)
            classification_counts = filtered_df['Classification'].value_counts()
            
            if 'Residential' in sum_df.columns:
                sum_df['Residential'] = (sum_df['Residential'] / classification_counts['Residential']) * 100
            if 'Commercial' in sum_df.columns:
                sum_df['Commercial'] = (sum_df['Commercial'] / classification_counts['Commercial']) * 100
            
            # Plot the results for CWSU-8
            if not sum_df.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Custom colors
                colors = []
                if 'Residential' in sum_df.columns:
                    colors.append('#b3e9c7')  # Green
                if 'Commercial' in sum_df.columns:
                    colors.append('#8367C7')  # Purple
                
                sum_df.plot(kind='bar', color=colors, ax=ax, width=0.8)
                
                # Formatting
                ax.set_title('Monthly Cost Distribution by Classification')
                ax.set_ylabel('Affirmative Responses (%)')
                ax.set_xlabel('Cost Range')
                ax.set_ylim(0, 100)
                
                # Set shorter x-axis labels
                ax.set_xticklabels([cost_mapping[label] for label in sum_df.index], rotation=45)
                
                # Add percentages on bars
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.1f}%', 
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', 
                                xytext=(0, 5), 
                                textcoords='offset points',
                                fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No data available for selected filters in CWSU-8")
        else:
            st.warning("No CWSU-8 monthly cost data found in the uploaded file")



        # CWSU-9 Tap Water Issues Analysis
        st.subheader("CWSU-9 Tap Water Issues Analysis")

        # Convert relevant columns to numeric and handle boolean values
        filtered_df.iloc[:, 50:61] = filtered_df.iloc[:, 50:61].apply(pd.to_numeric, errors='coerce')
        filtered_df.iloc[:, 50:61] = filtered_df.iloc[:, 50:61].replace({True: 1, False: 0}).astype(float)

        # Select target column (index 51) and feature columns (index 52 to 60)
        target_col = filtered_df.columns[51]
        feature_cols = filtered_df.columns[52:61]

        # Count occurrences of 1 and not 1 in the target column
        count_of_1 = (filtered_df[target_col] == 1).sum()
        count_of_other = (filtered_df[target_col] != 1).sum()

        # Create a DataFrame for the pie chart
        pie_data = pd.DataFrame({
            'Count': [count_of_1, count_of_other],
            'Category': ['Yes', 'No']
        })

        # Calculate correlation
        correlation_results = filtered_df[[target_col] + list(feature_cols)].corr()[target_col].drop(target_col)

        # Create horizontal layout figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Pie chart (left side)
        ax1.pie(pie_data['Count'], 
                labels=pie_data['Category'], 
                autopct='%1.1f%%', 
                colors=['#8367c7', '#b3e9c7'],
                startangle=90,
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        ax1.set_title('Presence of Tap Water Issues', pad=20)

        # Horizontal bar plot (right side)
        sns.barplot(x=correlation_results.values,
                   y=correlation_results.index,
                   palette='ch:s=.25,rot=-.25',
                   ax=ax2)
        ax2.set_title('Correlation with Contributing Factors', pad=20)
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_ylabel('')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add correlation values on bars
        for i, v in enumerate(correlation_results.values):
            ax2.text(v, i, f'{v:.2f}', 
                    color='black', 
                    ha='left' if v < 0 else 'right', 
                    va='center')

        # Adjust layout
        plt.tight_layout()
        st.pyplot(fig)



        # CWSU-10 Deep Well Issues Analysis
        st.subheader("CWSU-10 Deep Well Issues Analysis")

        # Convert relevant columns to numeric and handle boolean values
        filtered_df.iloc[:, 62:73] = filtered_df.iloc[:, 62:73].apply(pd.to_numeric, errors='coerce')
        filtered_df.iloc[:, 62:73] = filtered_df.iloc[:, 62:73].replace({True: 1, False: 0}).astype(float)

        # Select target column (index 62) and feature columns (index 63 to 72)
        target_col_dw = filtered_df.columns[62]  # Deep well target column
        feature_cols_dw = filtered_df.columns[63:73]

        # Count occurrences of 1 and not 1 in the target column
        count_of_1_dw = (filtered_df[target_col_dw] == 1).sum()
        count_of_other_dw = (filtered_df[target_col_dw] != 1).sum()

        # Create a DataFrame for the pie chart
        pie_data_dw = pd.DataFrame({
            'Count': [count_of_1_dw, count_of_other_dw],
            'Category': ['Yes', 'No']
        })

        # Calculate correlation
        correlation_results_dw = filtered_df[[target_col_dw] + list(feature_cols_dw)].corr()[target_col_dw].drop(target_col_dw)

        # Create side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Pie chart (left side)
        axes[0].pie(pie_data_dw['Count'], 
                    labels=pie_data_dw['Category'], 
                    autopct='%1.1f%%', 
                    colors=["#8367c7", "#b3e9c7"],
                    startangle=140)
        axes[0].set_title('Presence of Deep Well Issues', pad=20)
        axes[0].axis('equal')

        # Horizontal bar plot (right side)
        sns.barplot(x=correlation_results_dw.values,
                    y=correlation_results_dw.index,
                    palette="ch:s=.25,rot=-.25",
                    ax=axes[1])
        axes[1].set_title('Correlation with Contributing Factors', pad=20)
        axes[1].set_xlabel('Correlation Coefficient')
        axes[1].set_ylabel('')
        axes[1].grid(axis='x', alpha=0.3)

        # Add correlation values on bars
        for i, v in enumerate(correlation_results_dw.values):
            axes[1].text(v, i, f'{v:.2f}',
                        color='black',
                        ha='left' if v < 0 else 'right',
                        va='center')

        plt.tight_layout()
        st.pyplot(fig)




        # CWSU-11 Truck Delivery Issues Analysis
        st.subheader("CWSU-11 Truck Delivery Issues Analysis")

        # Convert relevant columns to numeric and handle boolean values
        filtered_df.iloc[:, 73:83] = filtered_df.iloc[:, 73:83].apply(pd.to_numeric, errors='coerce')
        filtered_df.iloc[:, 73:83] = filtered_df.iloc[:, 73:83].replace({True: 1, False: 0}).astype(float)

        # Select target column (index 73) and feature columns (index 74 to 82)
        target_col_td = filtered_df.columns[73]  # AWTP-14 - Factors affecting Alt Sources - Reliability (consistent supply)
        feature_cols_td = filtered_df.columns[74:83]

        # Count occurrences of 1 and not 1 in the target column
        count_of_1_td = (filtered_df[target_col_td] == 1).sum()
        count_of_other_td = (filtered_df[target_col_td] != 1).sum()

        # Create a DataFrame for the pie chart
        pie_data_td = pd.DataFrame({
            'Count': [count_of_1_td, count_of_other_td],
            'Category': ['Yes', 'No']
        })

        # Calculate correlation
        correlation_results_td = filtered_df[[target_col_td] + list(feature_cols_td)].corr()[target_col_td].drop(target_col_td)

        # Create side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Pie chart (left side)
        axes[0].pie(pie_data_td['Count'], 
                    labels=pie_data_td['Category'], 
                    autopct='%1.1f%%', 
                    colors=["#8367c7", "#b3e9c7"],
                    startangle=140)
        axes[0].set_title('Distribution of Responses: Presence of Truck Delivery Issues', pad=20)
        axes[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Correlation bar plot (right side)
        sns.barplot(x=correlation_results_td.values, 
                    y=correlation_results_td.index, 
                    palette="ch:s=.25,rot=-.25", 
                    ax=axes[1])
        axes[1].set_title('Factors Correlating to Truck Delivery Issues', pad=20)
        axes[1].set_xlabel('Correlation Coefficient')
        axes[1].set_ylabel('Feature Column')
        axes[1].grid(axis='x', alpha=0.3)

        # Add correlation values on bars
        for i, v in enumerate(correlation_results_td.values):
            axes[1].text(v, i, f'{v:.2f}',
                        color='black',
                        ha='left' if v < 0 else 'right',
                        va='center')

        # Adjust layout
        plt.tight_layout()
        st.pyplot(fig)


        # CWSU-12 Bottled Water Issues Analysis
        st.subheader("CWSU-12 Bottled Water Issues Analysis")

        # Convert relevant columns to numeric and handle boolean values
        filtered_df.iloc[:, 84:94] = filtered_df.iloc[:, 84:94].apply(pd.to_numeric, errors='coerce')
        filtered_df.iloc[:, 84:94] = filtered_df.iloc[:, 84:94].replace({True: 1, False: 0}).astype(float)

        # Select target column (index 84) and feature columns (index 85 to 93)
        target_col_bw = filtered_df.columns[84]  # AWTP-14 - Factors affecting Alt Sources - Water Quality (Clean and safe)
        feature_cols_bw = filtered_df.columns[85:94]

        # Count occurrences of 1 and not 1 in the target column
        count_of_1_bw = (filtered_df[target_col_bw] == 1).sum()
        count_of_other_bw = (filtered_df[target_col_bw] != 1).sum()

        # Create a DataFrame for the pie chart
        pie_data_bw = pd.DataFrame({
            'Count': [count_of_1_bw, count_of_other_bw],
            'Category': ['Yes', 'No']
        })

        # Calculate correlation
        correlation_results_bw = filtered_df[[target_col_bw] + list(feature_cols_bw)].corr()[target_col_bw].drop(target_col_bw)

        # Create side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Pie chart (left side)
        axes[0].pie(pie_data_bw['Count'], 
                    labels=pie_data_bw['Category'], 
                    autopct='%1.1f%%', 
                    colors=["#8367c7", "#b3e9c7"],
                    startangle=140)
        axes[0].set_title('Distribution of Responses: Presence of Bottled Water Issues', pad=20)
        axes[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Correlation bar plot (right side)
        sns.barplot(x=correlation_results_bw.values, 
                    y=correlation_results_bw.index, 
                    palette="ch:s=.25,rot=-.25", 
                    ax=axes[1])
        axes[1].set_title('Factors Correlating to Bottled Water Issues', pad=20)
        axes[1].set_xlabel('Correlation Coefficient')
        axes[1].set_ylabel('Feature Column')
        axes[1].grid(axis='x', alpha=0.3)

        # Add correlation values on bars
        for i, v in enumerate(correlation_results_bw.values):
            axes[1].text(v, i, f'{v:.2f}',
                        color='black',
                        ha='left' if v < 0 else 'right',
                        va='center')


        # Adjust layout
        plt.tight_layout()
        st.pyplot(fig)

        # CWSU-13 Water Supply Interruptions
        st.subheader("CWSU-13 Water Supply Interruptions")

        # Convert relevant columns to numeric, coercing errors to NaN
        filtered_df.iloc[:, 96:101] = filtered_df.iloc[:, 96:101].apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values in the specified columns
        filtered_df = filtered_df.dropna(subset=filtered_df.columns[96:101])

        # Convert boolean values to integers
        filtered_df.iloc[:, 96:101] = filtered_df.iloc[:, 96:101].replace({True: 1, False: 0}).astype(int)

        # Calculate response counts
        response_counts = filtered_df.iloc[:, 96:101].apply(pd.Series.value_counts).T.fillna(0)
        response_counts.columns = ['No', 'Yes']

        # Label mapping
        labels = [
            'Continuous Supply', 
            '1x Weekly Interruption',
            '2x Weekly Interruption',
            '3x Weekly Interruption', 
            'Other Interruptions'
        ]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Yes responses (bottom) in green
        ax.bar(labels, response_counts['Yes'], color='#b3e9c7', label='Interruption Occurs')
        
        # Plot No responses (top) in purple
        ax.bar(labels, response_counts['No'], bottom=response_counts['Yes'], 
              color='#8367c7', label='No Interruption')

        # Add value labels
        for i, (yes, no) in enumerate(zip(response_counts['Yes'], response_counts['No'])):
            ax.text(i, yes / 2, f'{int(yes)}', ha='center', va='center', color='black', fontweight='bold')
            ax.text(i, yes + no / 2, f'{int(no)}', ha='center', va='center', color='black', fontweight='bold')

        # Formatting
        ax.set_title('Water Supply Interruption Frequency', pad=15)
        ax.set_ylabel('Number of Households')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)



        # AWTP-14 Alternative Water Source Factors
        st.subheader("AWTP-14 Alternative Water Source Factors")

        # Select columns 102 to 107
        cols_to_plot = filtered_df.columns[102:107]

        # Clean and convert data
        filtered_df[cols_to_plot] = filtered_df[cols_to_plot].apply(pd.to_numeric, errors='coerce')
        filtered_df = filtered_df.dropna(subset=cols_to_plot)  # Remove rows with invalid data
        filtered_df[cols_to_plot] = filtered_df[cols_to_plot].astype(int)  # Force integer conversion

        # Calculate percentage of "Yes" (1) responses
        responses_pct = (filtered_df[cols_to_plot] == 1).mean() * 100

        # Create and style the bar graph
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(responses_pct)), responses_pct, color='#8367c7')

        # Formatting
        ax.set_title('Key Factors in Alternative Water Sourcing', pad=15)
        ax.set_ylabel('Affirmative Responses (%)')
        ax.set_xticks(range(len(responses_pct)))
        ax.set_xticklabels([
            'Cost-effectiveness', 
            'Reliability', 
            'Water Quality', 
            'Customer Service', 
            'Others'
        ], rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Add value labels
        for i, pct in enumerate(responses_pct):
            ax.text(i, pct, f"{pct:.1f}%", 
                   ha='center', va='bottom',
                   fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)


        # AWTP-15 Desalination Awareness and Willingness
        st.subheader("AWTP-15 Desalination Awareness and Willingness")

        # Select columns 108 and 110 (using 0-based indexing, so 107 and 109)
        cols_to_count = filtered_df.columns[[108, 110]]
        data_to_plot = filtered_df[['Classification'] + cols_to_count.tolist()].copy()

        # Rename columns for easier plotting
        data_to_plot.rename(columns={
            filtered_df.columns[108]: 'Awareness',
            filtered_df.columns[110]: 'Willingness'
        }, inplace=True)

        # Melt the DataFrame to long format for easier plotting
        melted_data = data_to_plot.melt(id_vars='Classification', var_name='Metric', value_name='Value')

        # Convert 'Value' column to numeric, coercing errors
        melted_data['Value'] = pd.to_numeric(melted_data['Value'], errors='coerce')

        # Filter for rows where the value is 1 (assuming 1 represents "Yes" or affirmation)
        yes_responses = melted_data[melted_data['Value'] == 1]

        # Calculate the percentage of "1" responses for each combination of Classification and Metric
        percentage_counts = yes_responses.groupby(['Classification', 'Metric']).size() / melted_data.groupby(['Classification', 'Metric']).size() * 100
        percentage_counts = percentage_counts.reset_index(name='Percentage')

        # Pivot the data back to wide format for grouped bar chart
        percentage_pivot = percentage_counts.pivot(index='Metric', columns='Classification', values='Percentage')

        # Ensure both Residential and Commercial columns exist, even if there are no '1's in one
        all_metrics = ['Awareness', 'Willingness']
        all_classifications = ['Residential', 'Commercial']
        percentage_pivot = percentage_pivot.reindex(all_metrics).reindex(columns=all_classifications).fillna(0)

        # Create a grouped bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        percentage_pivot.plot(kind='bar', color=['#b3e9c7', '#8367c7'], width=0.6, ax=ax)

        # Formatting
        ax.set_title('Desalination Awareness and Willingness by Classification', pad=20)
        ax.set_ylabel('Percentage of Affirmative Responses (%)')
        ax.set_xlabel('Metric')
        ax.set_xticks(range(len(all_metrics)))
        ax.set_xticklabels(all_metrics, rotation=0, ha='center')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(title='Classification')

        # Add percentage labels on top of bars
        for container in ax.containers:
            for rect in container.get_children():
                height = rect.get_height()  # Get the height of the rectangle
                if height > 0:  # Only label positive percentages
                    ax.annotate(f'{height:.2f}%',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

        plt.tight_layout()
        st.pyplot(fig)


        # AWTP-17 Desalinated Premium Pay
        st.subheader("AWTP-17 Desalinated Premium Pay")

        # Data for the first group (Desalinated Premium Pay - Yes, No, Depends)
        premium_pay_data_1 = filtered_df[['AWTP-17 - Desalinated Premium Pay - Yes',
                                            'AWTP-17 - Desalinated Premium Pay - No',
                                            'AWTP-17 - Desalinated Premium Pay - Depends on the cost difference']].fillna(0).sum()

        labels_1 = ['Yes', 'No', 'Depends on the cost difference']
        colors_1 = ["#8367c7", "#b3e9c7", "#E8B5B3"]  # Example colors

        # Data for the second group (Desalinated Premium Pay - Fixed tariff, Tiered pricing, Pay per use, Seasonal)
        premium_pay_data_2 = filtered_df[['AWTP-17 - Desalinated Premium Pay - Fixed tariff',
                                            'AWTP-17 - Desalinated Premium Pay - Tiered pricing',
                                            'AWTP-17 - Desalinated Premium Pay - Pay per use',
                                            'AWTP-17 - Desalinated Premium Pay - Seasonal']].fillna(0).sum()

        labels_2 = ['Fixed tariff', 'Tiered pricing', 'Pay per use', 'Seasonal']
        colors_2 = ["#E0AEAC", "#BE9392", "#9C7978", "#7A5F5E"]  # Example colors

        # Create separate pie charts
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Pie Chart 1
        axes[0].pie(premium_pay_data_1, labels=labels_1, autopct='%1.1f%%', colors=colors_1, startangle=140, labeldistance=1.05)
        axes[0].set_title('AWTP-17 - Desalinated Premium Pay: Acceptance')
        axes[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Pie Chart 2
        axes[1].pie(premium_pay_data_2, labels=labels_2, autopct='%1.1f%%', colors=colors_2, startangle=140, labeldistance=1.05)
        axes[1].set_title('AWTP-17 - Desalinated Premium Pay: Pricing Models Preference')
        axes[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Display the plots
        st.pyplot(fig)







     # Assuming filtered_df is your DataFrame containing the relevant data
        st.subheader("Operating / Peak Hours")
        if 'GI - Operating Hours - Start' in filtered_df.columns and 'GI - Operating Hours - End' in filtered_df.columns:
            # Convert to datetime and handle errors
            filtered_df['GI - Operating Hours - Start'] = pd.to_datetime(filtered_df['GI - Operating Hours - Start'], errors='coerce')
            filtered_df['GI - Operating Hours - End'] = pd.to_datetime(filtered_df['GI - Operating Hours - End'], errors='coerce')

            # Filter out rows with invalid dates
            filtered_df = filtered_df[
                filtered_df['GI - Operating Hours - Start'].notnull() &
                filtered_df['GI - Operating Hours - End'].notnull()
            ].copy()

            # Extract hours from the valid datetime columns
            filtered_df['Start Hour'] = filtered_df['GI - Operating Hours - Start'].dt.hour
            filtered_df['End Hour'] = filtered_df['GI - Operating Hours - End'].dt.hour

            # Create figure - COMPLETELY remove y-axis elements
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12),
                                        gridspec_kw={'height_ratios': [1, 0.4]})

            # Plot 1: Operating hours - FINAL clean version
            for i, (start, end) in enumerate(zip(filtered_df['Start Hour'], filtered_df['End Hour'])):
                ax1.plot([start + 0.25, end + 0.25], [i] * 2,
                        color='#b3e9c7', linewidth=2.5,
                        zorder=2)  # Ensure lines appear above grid

            # CRITICAL: Remove ALL y-axis elements
            ax1.set(xlim=(0, 24),
                    ylim=(-0.5, len(filtered_df)-0.5),
                    xticks=range(25),
                    title="Operating Hours")
            ax1.set_xticklabels([f"{h:02d}" for h in range(25)], rotation=45)
            ax1.set_yticklabels([])  # THIS removes all y-axis labels
            ax1.set_yticks([])       # THIS removes tick marks
            ax1.spines['left'].set_visible(False)  # Removes the spine
            ax1.spines[['right', 'top']].set_visible(False)
            ax1.grid(axis='x', linestyle='--', alpha=0.3)

            # Plot 2: Density plot
            hour_counts = [(filtered_df['Start Hour'] <= h) & (filtered_df['End Hour'] > h) for h in range(24)]
            percent_active = [round(count.sum() / len(filtered_df) * 100, 1) if len(filtered_df) > 0 else 0
                            for count in hour_counts]

            for h in range(24):
                alpha = min(percent_active[h]/100*0.8 + 0.2, 0.9)
                ax2.axvspan(h, h+1, color='#b3e9c7', alpha=alpha)
                if percent_active[h] >= 1:
                    ax2.text(h+0.5, 0.5, f"{percent_active[h]}%",
                            ha='center', va='center', color='black',
                            fontweight='bold', fontsize=9)

            ax2.set(xticks=np.arange(0.5, 24.5, 1),
                xlim=(0, 24),
                title='Hourly Operation Density (%)')
            ax2.set_xticklabels([f"{h:02d}" for h in range(24)], rotation=45)
            ax2.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
            ax2.set_yticks([])

            # Display the plots in Streamlit
            plt.tight_layout()
            st.pyplot(fig)
















    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.warning("Please upload a CSV file to analyze")
