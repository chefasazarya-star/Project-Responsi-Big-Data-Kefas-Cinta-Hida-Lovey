import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import untuk machine learning - PASTIKAN SUDAH DIINSTALL
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.sidebar.warning("⚠️ scikit-learn not installed. Some features disabled.")

# Import untuk statistical analysis
try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.sidebar.warning("⚠️ scipy not installed. Some statistical features disabled.")

# Set page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Judul aplikasi
st.title("🚀 Sales Data Analysis Dashboard")
st.markdown("**Analisis Data Penjualan Menggunakan PySpark untuk Mengidentifikasi Pola Penjualan Global**")
st.markdown("---")

# Sidebar untuk navigasi
st.sidebar.title("🎛️ Navigation")
page = st.sidebar.radio("Pilih Halaman:", [
    "📈 Dashboard Overview",
    "🏆 Produk Terlaris",
    "💰 Revenue Analysis",
    "🌍 Geographic Analysis",
    "📅 Monthly Trends",
    "👥 Top Customers",
    "🎯 Customer Segmentation",
    "🔍 EDA & Data Quality"
])


# FUNGSI UNTUK LOAD DATA ASLI
@st.cache_data
def load_real_data():
    """Load your actual CSV data"""
    try:
        # PASTIKAN NAMA FILE INI SAMA DENGAN FILE CSV KAMU
        df = pd.read_csv("part-00000-b2620b1f-8581-4e7f-aab4-f4aeaf8e03ba-c000.csv")
        st.sidebar.success("✅ Data asli berhasil diload!")
        return df
    except FileNotFoundError:
        st.sidebar.error("❌ File data asli tidak ditemukan! Pakai sample data.")
        return load_sample_data()
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {e}")
        return load_sample_data()


# Sample data untuk fallback
def load_sample_data():
    """Load sample data jika data asli tidak ada"""
    n_records = 100
    product_lines = ['Classic Cars', 'Vintage Cars', 'Motorcycles', 'Planes', 'Trucks and Buses', 'Ships', 'Trains']
    countries = ['USA', 'Australia', 'France', 'Germany', 'UK', 'Japan', 'Canada']
    customers = ['Customer A', 'Customer B', 'Customer C', 'Customer D', 'Customer E']

    np.random.seed(42)

    data = {
        'PRODUCTLINE': np.random.choice(product_lines, n_records),
        'QUANTITYORDERED': np.random.randint(10, 100, n_records),
        'TOTAL_REVENUE': np.random.randint(1000, 50000, n_records),
        'COUNTRY': np.random.choice(countries, n_records),
        'MONTH': np.random.randint(1, 13, n_records),
        'YEAR': np.random.choice([2023, 2024], n_records),
        'CUSTOMERNAME': np.random.choice(customers, n_records)
    }

    return pd.DataFrame(data)


# LOAD DATA - COBA DATA ASLI, JIKA GAGAL PAKAI SAMPLE
df = load_real_data()

# Tampilkan info data di sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Data Info")
    st.write(f"Jumlah data: {len(df):,} records")
    st.write(f"Kolom: {len(df.columns)}")
    st.write(f"Size: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

# ============================================================================
# 1. DASHBOARD OVERVIEW
# ============================================================================
if page == "📈 Dashboard Overview":
    st.header("📊 Dashboard Overview")

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_revenue = df['TOTAL_REVENUE'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")

    with col2:
        total_quantity = df['QUANTITYORDERED'].sum()
        st.metric("Total Quantity Sold", f"{total_quantity:,.0f}")

    with col3:
        unique_products = df['PRODUCTLINE'].nunique()
        st.metric("Product Categories", unique_products)

    with col4:
        unique_countries = df['COUNTRY'].nunique()
        st.metric("Countries", unique_countries)

    st.markdown("---")

    # Quick charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Sales by Product Line")
        product_sales = df.groupby('PRODUCTLINE')['TOTAL_REVENUE'].sum().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=product_sales.values, y=product_sales.index, ax=ax, palette="viridis")
        ax.set_title("Revenue by Product Line", fontweight='bold')
        ax.set_xlabel("Total Revenue ($)")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("🌍 Sales by Country")
        country_sales = df.groupby('COUNTRY')['TOTAL_REVENUE'].sum().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=country_sales.values, y=country_sales.index, ax=ax, palette="magma")
        ax.set_title("Revenue by Country", fontweight='bold')
        ax.set_xlabel("Total Revenue ($)")
        plt.tight_layout()
        st.pyplot(fig)

    # Additional insights
    st.markdown("---")
    st.subheader("💡 Key Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        top_product = df.groupby('PRODUCTLINE')['TOTAL_REVENUE'].sum().idxmax()
        st.info(f"**Top Product**: {top_product}")

    with col2:
        top_country = df.groupby('COUNTRY')['TOTAL_REVENUE'].sum().idxmax()
        st.info(f"**Top Country**: {top_country}")

    with col3:
        avg_order_value = df['TOTAL_REVENUE'].mean()
        st.info(f"**Avg Order Value**: ${avg_order_value:,.0f}")

# ============================================================================
# 2. PRODUK TERLARIS
# ============================================================================
elif page == "🏆 Produk Terlaris":
    st.header("🏆 Produk Terlaris Analysis")

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        sort_by = st.selectbox("Sort by:", ["Quantity", "Revenue"])

    with col2:
        top_n = st.slider("Show top:", 3, 10, 5)

    # Data processing
    if sort_by == "Quantity":
        product_data = df.groupby('PRODUCTLINE')['QUANTITYORDERED'].sum().nlargest(top_n)
        title = f"Top {top_n} Products by Quantity Sold"
        xlabel = "Quantity Sold"
    else:
        product_data = df.groupby('PRODUCTLINE')['TOTAL_REVENUE'].sum().nlargest(top_n)
        title = f"Top {top_n} Products by Revenue"
        xlabel = "Total Revenue ($)"

    # Visualization
    if not product_data.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(product_data)))

        bars = ax.barh(product_data.index, product_data.values, color=colors)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel)

        # Add value labels on bars
        for bar, value in zip(bars, product_data.values):
            width = bar.get_width()
            ax.text(width + product_data.values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{value:,.0f}', va='center', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        # Data table
        st.subheader("📋 Data Table")
        display_df = pd.DataFrame({
            'Product Line': product_data.index,
            'Value': product_data.values
        }).reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True)

        # Insights
        st.subheader("💡 Business Insights")
        top_product = product_data.index[0]
        top_value = product_data.values[0]

        if sort_by == "Revenue":
            st.success(f"**{top_product}** adalah produk dengan revenue tertinggi: **${top_value:,.0f}**")
            st.write("✅ **Rekomendasi**: Fokus pada pengembangan dan promosi produk ini")
        else:
            st.success(f"**{top_product}** adalah produk dengan quantity terjual tertinggi: **{top_value:,.0f} units**")
            st.write("✅ **Rekomendasi**: Optimalkan stok dan distribusi untuk produk ini")

    else:
        st.warning("No product data available")

# ============================================================================
# 3. REVENUE ANALYSIS
# ============================================================================
elif page == "💰 Revenue Analysis":
    st.header("💰 Revenue Analysis")

    # Revenue distribution
    st.subheader("Revenue Distribution by Product Line")

    # Pie chart
    revenue_by_product = df.groupby('PRODUCTLINE')['TOTAL_REVENUE'].sum()

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax1.pie(revenue_by_product.values,
                                           labels=revenue_by_product.index,
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           colors=plt.cm.Set3(np.linspace(0, 1, len(revenue_by_product))))
        ax1.set_title('Revenue Distribution (%)', fontweight='bold')
        plt.setp(autotexts, size=10, weight="bold")
        st.pyplot(fig1)

    with col2:
        # Bar chart
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        revenue_by_product_sorted = revenue_by_product.sort_values(ascending=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(revenue_by_product_sorted)))

        bars = ax2.barh(revenue_by_product_sorted.index, revenue_by_product_sorted.values, color=colors)
        ax2.set_title('Revenue by Product Line ($)', fontweight='bold')
        ax2.set_xlabel('Total Revenue ($)')

        # Add value labels
        for bar, value in zip(bars, revenue_by_product_sorted.values):
            ax2.text(bar.get_width() + revenue_by_product_sorted.values.max() * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f'${value:,.0f}', va='center', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig2)

    # Revenue statistics
    st.subheader("📊 Revenue Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_revenue = df['TOTAL_REVENUE'].mean()
        st.metric("Average Revenue", f"${avg_revenue:,.2f}")

    with col2:
        max_revenue = df['TOTAL_REVENUE'].max()
        st.metric("Highest Revenue", f"${max_revenue:,.0f}")

    with col3:
        min_revenue = df['TOTAL_REVENUE'].min()
        st.metric("Lowest Revenue", f"${min_revenue:,.0f}")

    with col4:
        total_revenue = df['TOTAL_REVENUE'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")

# ============================================================================
# 4. GEOGRAPHIC ANALYSIS
# ============================================================================
elif page == "🌍 Geographic Analysis":
    st.header("🌍 Geographic Analysis")

    # Country selection
    selected_country = st.selectbox("Select Country:", sorted(df['COUNTRY'].unique()))

    # Filter data
    country_data = df[df['COUNTRY'] == selected_country]

    if not country_data.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📈 Performance in {selected_country}")

            # Metrics
            country_revenue = country_data['TOTAL_REVENUE'].sum()
            country_quantity = country_data['QUANTITYORDERED'].sum()
            country_orders = len(country_data)
            avg_order_value = country_data['TOTAL_REVENUE'].mean()

            st.metric("Total Revenue", f"${country_revenue:,.0f}")
            st.metric("Total Quantity", f"{country_quantity:,.0f}")
            st.metric("Number of Orders", country_orders)
            st.metric("Average Order Value", f"${avg_order_value:,.0f}")

            # Top products in country
            st.subheader("🏆 Top Products")
            top_products = country_data.groupby('PRODUCTLINE')['TOTAL_REVENUE'].sum().nlargest(3)

            for product, revenue in top_products.items():
                st.write(f"• **{product}**: ${revenue:,.0f}")

        with col2:
            st.subheader("📊 Product Distribution")

            fig, ax = plt.subplots(figsize=(10, 6))
            product_dist = country_data.groupby('PRODUCTLINE')['QUANTITYORDERED'].sum().sort_values(ascending=True)

            if not product_dist.empty:
                colors = plt.cm.viridis(np.linspace(0, 1, len(product_dist)))
                bars = ax.barh(product_dist.index, product_dist.values, color=colors)
                ax.set_title(f"Product Sales in {selected_country}", fontweight='bold')
                ax.set_xlabel("Quantity Sold")

                # Add value labels
                for bar, value in zip(bars, product_dist.values):
                    ax.text(bar.get_width() + product_dist.values.max() * 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f'{value:,.0f}', va='center', fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No product data available")

    else:
        st.warning("No data available for selected country")

# ============================================================================
# 5. MONTHLY TRENDS
# ============================================================================
elif page == "📅 Monthly Trends":
    st.header("📅 Monthly Trends Analysis")

    # Create monthly data
    monthly_data = df.groupby(['YEAR', 'MONTH']).agg({
        'TOTAL_REVENUE': 'sum',
        'QUANTITYORDERED': 'sum',
        'ORDERNUMBER': 'count'
    }).reset_index()

    monthly_data.rename(columns={'ORDERNUMBER': 'ORDER_COUNT'}, inplace=True)

    # Create date column for plotting
    monthly_data['DATE'] = pd.to_datetime(
        monthly_data['YEAR'].astype(str) + '-' + monthly_data['MONTH'].astype(str) + '-01'
    )

    # Year selection
    selected_year = st.selectbox("Select Year:", sorted(monthly_data['YEAR'].unique()))

    # Filter data
    yearly_data = monthly_data[monthly_data['YEAR'] == selected_year]

    # Plot
    st.subheader(f"📈 Monthly Trend for {selected_year}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Revenue trend
    ax1.plot(yearly_data['MONTH'], yearly_data['TOTAL_REVENUE'],
             marker='o', linewidth=2, markersize=8, label='Revenue', color='#FF6B6B')
    ax1.set_title(f"Monthly Revenue Trend - {selected_year}", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Revenue ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 13))

    # Quantity trend
    ax2.plot(yearly_data['MONTH'], yearly_data['QUANTITYORDERED'],
             marker='s', linewidth=2, markersize=6, label='Quantity', color='#4ECDC4')
    ax2.set_title(f"Monthly Quantity Trend - {selected_year}", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Quantity Sold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 13))

    plt.tight_layout()
    st.pyplot(fig)

    # Monthly statistics
    st.subheader("📊 Monthly Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        best_month_data = yearly_data.loc[yearly_data['TOTAL_REVENUE'].idxmax()]
        st.metric("Best Month (Revenue)",
                  f"Month {int(best_month_data['MONTH'])}",
                  f"${best_month_data['TOTAL_REVENUE']:,.0f}")

    with col2:
        avg_monthly = yearly_data['TOTAL_REVENUE'].mean()
        st.metric("Average Monthly Revenue", f"${avg_monthly:,.0f}")

    with col3:
        if len(yearly_data) > 1:
            growth = ((yearly_data['TOTAL_REVENUE'].iloc[-1] - yearly_data['TOTAL_REVENUE'].iloc[0]) /
                      yearly_data['TOTAL_REVENUE'].iloc[0] * 100)
            st.metric("Growth Rate", f"{growth:.1f}%")
        else:
            st.metric("Growth Rate", "N/A")

    with col4:
        total_yearly = yearly_data['TOTAL_REVENUE'].sum()
        st.metric(f"Total {selected_year} Revenue", f"${total_yearly:,.0f}")

# ============================================================================
# 6. TOP CUSTOMERS
# ============================================================================
elif page == "👥 Top Customers":
    st.header("👥 Top Customers Analysis")

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        top_n_customers = st.slider("Number of Top Customers:", 5, 20, 10)

    with col2:
        sort_by_cust = st.selectbox("Sort Customers by:", ["Revenue", "Quantity", "Order Count"])

    # Calculate customer metrics
    if sort_by_cust == "Revenue":
        customer_data = df.groupby('CUSTOMERNAME').agg({
            'TOTAL_REVENUE': 'sum',
            'QUANTITYORDERED': 'sum',
            'ORDERNUMBER': 'count'
        }).nlargest(top_n_customers, 'TOTAL_REVENUE')
        title = f"Top {top_n_customers} Customers by Revenue"
        xlabel = "Total Revenue ($)"
    elif sort_by_cust == "Quantity":
        customer_data = df.groupby('CUSTOMERNAME').agg({
            'TOTAL_REVENUE': 'sum',
            'QUANTITYORDERED': 'sum',
            'ORDERNUMBER': 'count'
        }).nlargest(top_n_customers, 'QUANTITYORDERED')
        title = f"Top {top_n_customers} Customers by Quantity"
        xlabel = "Total Quantity"
    else:
        customer_data = df.groupby('CUSTOMERNAME').agg({
            'TOTAL_REVENUE': 'sum',
            'QUANTITYORDERED': 'sum',
            'ORDERNUMBER': 'count'
        }).nlargest(top_n_customers, 'ORDERNUMBER')
        title = f"Top {top_n_customers} Customers by Order Count"
        xlabel = "Number of Orders"

    customer_data = customer_data.sort_values(customer_data.columns[0], ascending=True)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    if sort_by_cust == "Revenue":
        values = customer_data['TOTAL_REVENUE']
    elif sort_by_cust == "Quantity":
        values = customer_data['QUANTITYORDERED']
    else:
        values = customer_data['ORDERNUMBER']

    colors = plt.cm.plasma(np.linspace(0, 1, len(customer_data)))
    bars = ax.barh(customer_data.index, values, color=colors)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel)

    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{value:,.0f}', va='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    # Customer details table
    st.subheader("📋 Customer Details")
    customer_data_display = customer_data.rename(columns={
        'TOTAL_REVENUE': 'Total Revenue',
        'QUANTITYORDERED': 'Total Quantity',
        'ORDERNUMBER': 'Order Count'
    })
    st.dataframe(customer_data_display, use_container_width=True)

    # Customer insights
    st.subheader("💡 Customer Insights")

    top_customer = customer_data.index[0]
    top_customer_revenue = customer_data['TOTAL_REVENUE'].iloc[0]
    top_customer_quantity = customer_data['QUANTITYORDERED'].iloc[0]
    top_customer_orders = customer_data['ORDERNUMBER'].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Top Customer", top_customer)

    with col2:
        st.metric("Their Total Revenue", f"${top_customer_revenue:,.0f}")

    with col3:
        st.metric("Their Total Orders", top_customer_orders)

    st.success(f"""
    **Key Insight:** {top_customer} adalah customer paling berharga dengan kontribusi revenue sebesar **${top_customer_revenue:,.0f}**
    dari {top_customer_orders} orders dan {top_customer_quantity:,.0f} unit produk.
    """)

# 7. CUSTOMER SEGMENTATION (K-Means Clustering)
elif page == "🎯 Customer Segmentation":
    st.header("🎯 Customer Segmentation dengan K-Means Clustering")

    if not ML_AVAILABLE:
        st.error("""
        ❌ **scikit-learn not installed!**

        Please install required libraries by running in terminal:
        ```bash
        pip install scikit-learn scipy
        ```
        """)
        st.stop()

    st.info("""
    **Analisis Clustering** untuk mengelompokkan transaksi berdasarkan pola pembelian menggunakan algoritma K-Means.
    Hasil clustering membantu identifikasi segmen customer yang berbeda berdasarkan karakteristik pembelian.
    """)

    # Prepare data for clustering
    clustering_features = ['QUANTITYORDERED', 'PRICEEACH', 'TOTAL_REVENUE']
    clustering_data = df[clustering_features].dropna()

    if len(clustering_data) < 10:
        st.warning("Not enough data for clustering analysis. Need at least 10 records.")
        st.stop()

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # K-Means Configuration
    st.sidebar.subheader("🔧 Clustering Settings")
    n_clusters = st.sidebar.slider("Number of Clusters (k):", 2, 5, 3)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Add cluster labels to data
    clustering_data = clustering_data.copy()
    clustering_data['CLUSTER'] = cluster_labels

    # Calculate cluster characteristics for naming
    cluster_stats = clustering_data.groupby('CLUSTER').agg({
        'QUANTITYORDERED': 'mean',
        'PRICEEACH': 'mean',
        'TOTAL_REVENUE': 'mean'
    })

    # Dynamic cluster naming based on characteristics
    cluster_names = {}
    overall_qty_median = clustering_data['QUANTITYORDERED'].median()
    overall_price_median = clustering_data['PRICEEACH'].median()

    for cluster_id in range(n_clusters):
        stats = cluster_stats.loc[cluster_id]
        qty = stats['QUANTITYORDERED']
        price = stats['PRICEEACH']
        revenue = stats['TOTAL_REVENUE']

        if qty > overall_qty_median and price > overall_price_median:
            cluster_names[cluster_id] = 'High Value Bulk'
        elif qty < overall_qty_median and price > overall_price_median:
            cluster_names[cluster_id] = 'Premium Products'
        elif qty > overall_qty_median and price < overall_price_median:
            cluster_names[cluster_id] = 'Volume Drivers'
        else:
            cluster_names[cluster_id] = 'Standard Transactions'

    clustering_data['CLUSTER_NAME'] = clustering_data['CLUSTER'].map(cluster_names)

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)

    # Display clustering results
    st.subheader("📊 Clustering Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Silhouette Score", f"{silhouette_avg:.4f}")

    with col2:
        st.metric("Number of Clusters", n_clusters)

    with col3:
        st.metric("Total Transactions", len(clustering_data))

    with col4:
        cluster_counts = clustering_data['CLUSTER'].value_counts()
        st.metric("Largest Cluster", f"{cluster_counts.max():,}")

    # Silhouette Score Interpretation
    st.subheader("📈 Model Evaluation - Silhouette Score")

    if silhouette_avg >= 0.7:
        score_status = "Excellent"
        score_color = "green"
    elif silhouette_avg >= 0.5:
        score_status = "Good"
        score_color = "blue"
    elif silhouette_avg >= 0.25:
        score_status = "Fair"
        score_color = "orange"
    else:
        score_status = "Poor"
        score_color = "red"

    st.write(f"**Silhouette Score**: {silhouette_avg:.4f} - **{score_status}**")

    # Simple progress bar without color issues
    progress_value = min(silhouette_avg, 1.0)  # Ensure value doesn't exceed 1.0
    st.progress(float(progress_value))

    st.info("""
    **Interpretation Silhouette Score:**
    - **0.7 - 1.0**: Strong cluster structure
    - **0.5 - 0.7**: Reasonable cluster structure  
    - **0.25 - 0.5**: Weak cluster structure
    - **< 0.25**: No substantial cluster structure
    """)

    st.markdown("---")

    # Cluster Interpretation
    st.subheader("📊 Cluster Characteristics")

    # Calculate detailed cluster characteristics
    cluster_summary = clustering_data.groupby(['CLUSTER', 'CLUSTER_NAME']).agg({
        'QUANTITYORDERED': ['mean', 'std', 'count'],
        'PRICEEACH': ['mean', 'std'],
        'TOTAL_REVENUE': ['mean', 'sum', 'count']
    }).round(2)

    # Display cluster characteristics
    st.dataframe(cluster_summary, use_container_width=True)

    # Visualize clusters
    st.subheader("📈 Cluster Visualization")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Scatter plot: Quantity vs Price with clusters
    scatter = ax1.scatter(
        clustering_data['QUANTITYORDERED'],
        clustering_data['PRICEEACH'],
        c=clustering_data['CLUSTER'],
        cmap='viridis',
        alpha=0.6,
        s=clustering_data['TOTAL_REVENUE'] / 100
    )
    ax1.set_title('K-Means Clustering: Quantity vs Price', fontweight='bold')
    ax1.set_xlabel('Quantity Ordered')
    ax1.set_ylabel('Price Each ($)')
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    ax1.grid(True, alpha=0.3)

    # Bar chart: Cluster sizes
    cluster_sizes = clustering_data['CLUSTER'].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_sizes)))
    bars = ax2.bar(range(len(cluster_sizes)), cluster_sizes.values, color=colors)
    ax2.set_title('Cluster Sizes', fontweight='bold')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Transactions')
    ax2.set_xticks(range(len(cluster_sizes)))
    ax2.set_xticklabels([f'Cluster {i}' for i in cluster_sizes.index])

    # Add value labels on bars
    for bar, value in zip(bars, cluster_sizes.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + cluster_sizes.values.max() * 0.01,
                 f'{value:,}', ha='center', va='bottom', fontweight='bold')

    # Revenue by cluster
    revenue_by_cluster = clustering_data.groupby('CLUSTER')['TOTAL_REVENUE'].sum()
    ax3.bar(range(len(revenue_by_cluster)), revenue_by_cluster.values, color=colors)
    ax3.set_title('Total Revenue by Cluster', fontweight='bold')
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Total Revenue ($)')
    ax3.set_xticks(range(len(revenue_by_cluster)))
    ax3.set_xticklabels([f'Cluster {i}' for i in revenue_by_cluster.index])

    # Add value labels on bars
    for i, value in enumerate(revenue_by_cluster.values):
        ax3.text(i, value + revenue_by_cluster.values.max() * 0.01, f'${value:,.0f}',
                 ha='center', va='bottom', fontweight='bold')

    # Average metrics by cluster
    avg_quantity = clustering_data.groupby('CLUSTER')['QUANTITYORDERED'].mean()
    avg_price = clustering_data.groupby('CLUSTER')['PRICEEACH'].mean()

    x = np.arange(len(avg_quantity))
    width = 0.35

    bars1 = ax4.bar(x - width / 2, avg_quantity.values, width, label='Avg Quantity', alpha=0.8, color='skyblue')
    bars2 = ax4.bar(x + width / 2, avg_price.values, width, label='Avg Price', alpha=0.8, color='lightcoral')
    ax4.set_title('Average Metrics by Cluster', fontweight='bold')
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Value')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Cluster {i}' for i in avg_quantity.index])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars1, avg_quantity.values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + avg_quantity.values.max() * 0.01,
                 f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar, value in zip(bars2, avg_price.values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + avg_price.values.max() * 0.01,
                 f'${value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    # Business Insights
    st.markdown("---")
    st.subheader("💡 Business Insights dari Clustering")

    # Generate detailed insights for each cluster
    for cluster_id in sorted(clustering_data['CLUSTER'].unique()):
        cluster_data = clustering_data[clustering_data['CLUSTER'] == cluster_id]
        cluster_name = cluster_data['CLUSTER_NAME'].iloc[0]

        avg_quantity = cluster_data['QUANTITYORDERED'].mean()
        avg_price = cluster_data['PRICEEACH'].mean()
        avg_revenue = cluster_data['TOTAL_REVENUE'].mean()
        total_revenue = cluster_data['TOTAL_REVENUE'].sum()
        size = len(cluster_data)
        revenue_share = (total_revenue / clustering_data['TOTAL_REVENUE'].sum()) * 100

        st.write(f"### 🎯 {cluster_name} (Cluster {cluster_id})")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Avg Quantity", f"{avg_quantity:.1f}")
        with col2:
            st.metric("Avg Price", f"${avg_price:.1f}")
        with col3:
            st.metric("Cluster Size", f"{size:,}")
        with col4:
            st.metric("Revenue Share", f"{revenue_share:.1f}%")

        # Generate specific insights based on cluster type
        if cluster_name == 'Premium Products':
            st.success("""
            **Segment Characteristics**: 
            - High price per unit, low volume
            - Premium customer segment
            - High margin products

            **Recommended Strategy**: 
            - Focus on high margin maintenance
            - Personalized customer service
            - Exclusive loyalty programs
            - Premium product bundles
            """)

        elif cluster_name == 'High Value Bulk':
            st.success("""
            **Segment Characteristics**: 
            - High quantity, high price
            - Business/wholesale customers
            - Significant revenue contributors

            **Recommended Strategy**: 
            - Optimize supply chain for bulk orders
            - Volume-based discount strategies
            - Long-term relationship building
            """)

        elif cluster_name == 'Volume Drivers':
            st.success("""
            **Segment Characteristics**: 
            - High quantity, low price
            - Mass market products
            - High inventory turnover

            **Recommended Strategy**: 
            - Operational efficiency focus
            - Economies of scale
            - Market penetration
            """)

        else:  # Standard Transactions
            st.success("""
            **Segment Characteristics**: 
            - Moderate quantity and price
            - Regular customer base
            - Stable revenue stream

            **Recommended Strategy**: 
            - Standardized service delivery
            - Efficient transaction processing
            - Base profitability maintenance
            """)

        st.markdown("---")

# 8. EDA & DATA QUALITY
elif page == "🔍 EDA & Data Quality":
    st.header("🔍 Exploratory Data Analysis & Data Quality")

    st.info("""
    **Comprehensive Data Analysis** - Detailed examination of data characteristics, quality assessment, 
    and statistical profiling to ensure data reliability for analysis.
    """)

    # Data Overview Section
    st.subheader("📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        st.metric("Total Columns", len(df.columns))

    with col3:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", missing_values)

    with col4:
        duplicate_rows = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicate_rows)

    # Data Quality Assessment
    st.subheader("✅ Data Quality Assessment")

    quality_data = []
    for column in df.columns:
        missing = df[column].isnull().sum()
        missing_pct = (missing / len(df)) * 100
        dtype = df[column].dtype
        unique_count = df[column].nunique()

        quality_data.append({
            'Column': column,
            'Data Type': str(dtype),
            'Missing Values': missing,
            'Missing %': f"{missing_pct:.2f}%",
            'Unique Values': unique_count
        })

    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, use_container_width=True)

    # Data Quality Scoring
    st.subheader("📋 Data Quality Scorecard")

    total_columns = len(df.columns)
    perfect_columns = len([x for x in quality_data if x['Missing Values'] == 0])
    quality_score = (perfect_columns / total_columns) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Quality Score", f"{quality_score:.1f}%")

    with col2:
        st.metric("Perfect Columns", f"{perfect_columns}/{total_columns}")

    with col3:
        problematic_cols = total_columns - perfect_columns
        st.metric("Columns with Issues", problematic_cols)

    with col4:
        completeness_rate = ((len(df) - missing_values) / (len(df) * total_columns)) * 100
        st.metric("Data Completeness", f"{completeness_rate:.1f}%")

    # Statistical Summary
    st.subheader("📈 Statistical Summary - Numerical Columns")

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        st.dataframe(df[numeric_columns].describe(), use_container_width=True)

    # Categorical Analysis
    st.subheader("📊 Categorical Columns Analysis")

    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            with st.expander(f"📁 {col} - Value Distribution"):
                value_counts = df[col].value_counts().head(10)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

                # Horizontal bar chart
                bars = ax1.barh(value_counts.index, value_counts.values, color='lightblue')
                ax1.set_title(f'Top 10 Values in {col}', fontweight='bold')
                ax1.set_xlabel('Count')

                for bar, value in zip(bars, value_counts.values):
                    ax1.text(bar.get_width() + value_counts.values.max() * 0.01,
                             bar.get_y() + bar.get_height() / 2,
                             f'{value:,}', va='center', fontweight='bold')

                # Pie chart for top 5
                if len(value_counts) >= 5:
                    top5 = value_counts.head(5)
                    ax2.pie(top5.values, labels=top5.index, autopct='%1.1f%%', startangle=90)
                    ax2.set_title(f'Top 5 Values Distribution - {col}')

                plt.tight_layout()
                st.pyplot(fig)

                # Value counts table
                st.dataframe(value_counts, use_container_width=True)

    # Distribution Analysis
    st.subheader("📊 Distribution Analysis - Numerical Variables")

    if len(numeric_columns) > 0:
        dist_column = st.selectbox("Select column for distribution analysis:",
                                   numeric_columns)

        if dist_column:
            # Create comprehensive distribution analysis
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            data_clean = df[dist_column].dropna()

            # Histogram with KDE
            ax1.hist(data_clean, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
            data_clean.plot.density(ax=ax1, color='red', linewidth=2)
            ax1.set_title(f'Distribution with KDE - {dist_column}', fontweight='bold')
            ax1.set_xlabel(dist_column)
            ax1.set_ylabel('Density')
            ax1.grid(True, alpha=0.3)
            ax1.legend(['KDE', 'Histogram'])

            # Box plot
            ax2.boxplot(data_clean)
            ax2.set_title(f'Box Plot - {dist_column}', fontweight='bold')
            ax2.set_ylabel(dist_column)
            ax2.grid(True, alpha=0.3)

            # QQ plot for normality check
            from scipy import stats

            stats.probplot(data_clean, dist="norm", plot=ax3)
            ax3.set_title(f'Q-Q Plot: {dist_column}', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # Cumulative distribution
            ax4.hist(data_clean, bins=30, alpha=0.7, color='lightgreen',
                     cumulative=True, density=True, edgecolor='black')
            ax4.set_title(f'Cumulative Distribution - {dist_column}', fontweight='bold')
            ax4.set_xlabel(dist_column)
            ax4.set_ylabel('Cumulative Probability')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # Distribution statistics
            st.subheader(f"📈 Statistical Properties - {dist_column}")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                skewness = data_clean.skew()
                st.metric("Skewness", f"{skewness:.2f}")
                st.caption(">0: Right skewed, <0: Left skewed")

            with col2:
                kurtosis = data_clean.kurtosis()
                st.metric("Kurtosis", f"{kurtosis:.2f}")
                st.caption(">3: Heavy tails, <3: Light tails")

            with col3:
                cv = (data_clean.std() / data_clean.mean()) * 100
                st.metric("Coefficient of Variation", f"{cv:.1f}%")
                st.caption("Variability relative to mean")

            with col4:
                outlier_threshold = data_clean.mean() + 3 * data_clean.std()
                outliers = len(data_clean[data_clean > outlier_threshold])
                st.metric("Potential Outliers", outliers)
                st.caption("Beyond 3 standard deviations")

    # Correlation Analysis
    st.subheader("🔗 Correlation Analysis")

    if len(numeric_columns) > 1:
        correlation_matrix = df[numeric_columns].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    ax=ax, mask=mask, square=True)
        ax.set_title('Correlation Matrix (Upper Triangle)', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'Variable 1': correlation_matrix.columns[i],
                        'Variable 2': correlation_matrix.columns[j],
                        'Correlation': f"{corr:.3f}",
                        'Strength': 'Very Strong' if abs(corr) > 0.9 else 'Strong'
                    })

        if strong_correlations:
            st.subheader("💡 Strong Correlations Found")
            strong_corr_df = pd.DataFrame(strong_correlations)
            st.dataframe(strong_corr_df, use_container_width=True)
        else:
            st.info("No very strong correlations (|r| > 0.7) found among numerical variables.")

    # Data Quality Recommendations
    st.markdown("---")
    st.subheader("🎯 Data Quality Recommendations & Action Plan")

    recommendations = []

    if missing_values > 0:
        recommendations.append("""
        🔸 **Handle Missing Values**: 
        - Implement data imputation strategies
        - Consider removal if missing < 5%
        - Document missing data patterns
        """)

    if duplicate_rows > 0:
        recommendations.append("""
        🔸 **Remove Duplicate Records**: 
        - Clean duplicate entries to prevent analysis bias
        - Investigate source of duplication
        - Implement data validation rules
        """)

    if quality_score < 90:
        recommendations.append("""
        🔸 **Improve Data Collection Process**: 
        - Enhance data entry validation
        - Implement required field constraints
        - Provide data quality training
        """)

    # Check for potential data issues
    for col in numeric_columns:
        if df[col].min() < 0 and col not in ['YEAR', 'MONTH']:
            recommendations.append(f"""
            🔸 **Verify Negative Values in {col}**: 
            - Check data entry accuracy
            - Validate business logic for negative values
            - Consider data transformation if appropriate
            """)

    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("""
        ✅ **Excellent Data Quality**: 
        - No major data quality issues detected
        - Data is ready for advanced analytics
        - Maintain current data governance practices
        """)

# Footer dengan credit lengkap
st.markdown("---")
st.markdown("### 🎯 Sales Analytics Dashboard")
st.markdown("**Dibuat oleh:**")
st.markdown("- Kefas Azarya (235091000111005)")
st.markdown("- Cinta Amalia Putri (235091001111007)")
st.markdown("- Zahidah Najla Nur Afifah (235091007111002)")
st.markdown("- Shabrina Deslovey Asmara (235091007111008)")
st.markdown("**Program Studi Sarjana Ilmu Aktuaria**")
st.markdown("**Departemen Matematika - Fakultas MIPA**")
st.markdown("**Universitas Brawijaya**")
st.markdown("**2025**")
st.markdown("---")
st.markdown("**Analisis Data Penjualan Menggunakan PySpark untuk Mengidentifikasi Pola Penjualan Global**")