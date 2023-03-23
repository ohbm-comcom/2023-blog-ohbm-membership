# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # OHBM Membership over the years
#
# Code for the plotting functions used in the [OHBM Blogpost](https://www.ohbmbrainmappingblog.com/blog/introducing-ohbm-membership-membership-over-the-years) that introduces the new OHBM membership tier MEMBERSHIP+ and looks at OHBM’s membership data, reflecting on OHBM’s development from an annual meeting to a scientific society.

# %%
from watermark import watermark
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import PIL
import country_converter as coco
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
import io
import warnings
print(watermark(packages='pandas,numpy,plotly,matplotlib,PIL,country_converter,seaborn,statsmodels,watermark'))

# %%
def warp_continent_data_for_map(continent_data):
    """To create map using px.choropleth, there seems to be no way around specifying each country if one wants to plot
    data on a per-continent basis. This slightly inefficient function does this. First by associating each continent
    with all countries that are associated with a continent and second by creating duplicate rows for each conference /
    continent pair.

    Args:
        continent_data (pd.DataFrame): Pandas Dataframe having the data on a per conference basis.

    Returns:
       pd.DataFrame: Dataframe with repeated rows.
    """
    continent_data['Country'] = ''
    countries = pd.read_csv(coco.COUNTRY_DATA_FILE, sep='\t')[['name_short', 'continent', 'UNregion', 'ISO3']]

    continent_dict = {ii : [] for ii in ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']}

    for co in countries.iterrows():
        tmp_con = co[1]['continent']
        tmp_un = co[1]['UNregion']

        if tmp_con == 'America' and (tmp_un == 'Northern America' or tmp_un =='Caribbean'
                                     or tmp_un=='Central America'):
            con = 'North America'
        elif tmp_un == 'South America':
            con = 'South America'
        else:
            con = tmp_con

        if con in list(continent_dict.keys()):
            continent_dict[con].append(co[1]['name_short'])

    continent_data_map = continent_data.copy()

    for con in list(continent_dict.keys()):
        continent_data.loc[continent_data['Continent']==con, 'Country'] = ','.join(continent_dict[con])

    # based on the inefficient solution from here
    # https://stackoverflow.com/questions/45965128/duplicating-pandas-dataframe-rows-based-on-string-split-without-iteration
    map_data = continent_data.copy()
    map_data = map_data.reset_index()
    map_data = map_data.set_index(['index', 'Country'])

    df2 = map_data.iloc[:0]

    for index, row in map_data.iterrows():
        stgs = index[1].split(",")
        for s in stgs:
            row.name = (index[0], s)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The frame.append")
                df2 = df2.append(row)

    map_data = df2.reset_index().rename(columns={'level_1': 'Country'})

    return map_data
# %% [markdown]
# ### Function for figure 1.

# %%
def plot_members_by_continent(continent_data, conference_continent, sns_cols,
                         filename='Fig1_conference_continent.png'):

    continents = np.unique(continent_data['Continent'])

    fig, axes = plt.subplots(2, len(continents)//2, figsize=(40, 16))
    axes = axes.flatten()

    n_conferences = len(np.unique(continent_data.Conference))

    for nn, (ax, cn) in enumerate(zip(axes, continents)):
        tmp_cn = continent_data.query('Continent == @cn')

        # Colors for if conference is on continent
        clrs = [sns_cols[1] if (x == cn) else sns_cols[0] for x in conference_continent]

        # Grey backgrond for virtual conferences
        ax.axvspan(13.45, 15.55, facecolor='black', alpha=0.15)
        ax.bar(np.arange(n_conferences), tmp_cn.Members.values, color=clrs)
        ax.set_title(label=cn, fontdict={'fontsize': 36})

        if nn >= 3:
            ax.set_xticks(np.arange(n_conferences), tmp_cn.Conference.values)
            x_labels = [ii.split(' ', 1)[1] for ii in tmp_cn.Conference.values]
            ax.set_xticklabels(x_labels, rotation=90, fontdict={'size': 24,  'horizontalalignment': 'center'})
        else:
            ax.set_xticks(np.arange(tmp_cn.shape[0]))
            ax.set_xticklabels([''] * n_conferences)

        if cn in ['Europe', 'North America']:
            ax.set(ylim=[0, 3000])
        elif cn in ['Africa', 'South America']:
            ax.set(ylim=[0, 100])

        if nn in [0, 3]:
            ax.set_ylabel('Members', fontdict={'fontsize': 24})

        ax.set_yticklabels(ax.get_yticklabels(), fontdict={'size': 20})

    plt.suptitle('Conferences by Continent 2004 - 2022', fontsize=50)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
# %% [markdown]
# ### Function for figure 2.

# %%
def plot_animated_map(member_data, conference_coordinates, save_to_gif=True, file_name='Fig2_gif_map.png'):

    df = pd.DataFrame(conference_coordinates)
    conference_df = df.transpose().reset_index().rename(columns={0: 'lat', 1: 'long',
                                                                 'index': 'Conference',
                                                                 2: 'City'})
    conference_df['Size'] = 5
    conference_df.loc[conference_df.City == 'Virtual', 'Size'] = 0
    conference_df['Color'] = 200

    # taken from
    # https://stackoverflow.com/questions/55460434/how-to-export-save-an-animated-bubble-chart-made-with-plotly

    fig = px.choropleth(member_data, locations="Country",
                        color=member_data["Members"],
                        hover_name="Country",
                        locationmode="country names",
                        animation_frame='Conference',
                        range_color=[0, member_data['Members'].max()],
                        title='Conference',
                        color_continuous_scale=px.colors.sequential.deep)


    fig2 = px.scatter_geo(conference_df, lon='long', lat='lat',
                          animation_frame='Conference', text='City', size='Size',
                          size_max=8, title='Conference', opacity=1.0, color='Color',
                          color_continuous_scale=px.colors.sequential.gray_r,
                          color_discrete_sequence=px.colors.sequential.gray_r)
    fig.add_trace(fig2.data[0])

    fig.update_layout(margin=dict(l=20,r=0,b=0,t=40,pad=0),
                      paper_bgcolor="white", height=500, width=900,
                      font_size=14,
                      title = {'text': "OHBM Membership", 'y':0.96, 'x':0.5,
                               'xanchor': 'center', 'yanchor': 'top'})
    fig.update_layout(sliders=[{"currentvalue": {"prefix": ""},
                                'pad': {'b': 10, 't': 10},
                                "visible" :True}])

    fig['layout'].pop('updatemenus')

    frames = []

    for i, frame in enumerate(fig.frames):
        fig.frames[i].data += (fig2.frames[i].data[0],)
        fig.frames[i]['data'][1]['textposition'] = 'bottom center'
        fig.frames[i]['data'][1]['textfont'] = {'color':'black', 'size': 24}
        fig.frames[i]['data'][1]['marker']['symbol'] = 'octagon-dot'
        fig.frames[i]['data'][1]['marker']['line']['color'] = 'black'

    if save_to_gif:
        # generate images for each step in animation
        for s, fr in enumerate(fig.frames):
            # set main traces to appropriate traces within plotly frame
            fig.update(data=fr.data)
            # move slider to correct place
            fig.layout.sliders[0].update(active=s)
            # generate image of current state
            frames.append(PIL.Image.open(io.BytesIO(fig.to_image(format="png"))))

        # create animated GIF
        frames[0].save(file_name, save_all=True, append_images=frames[1:],
                       optimize=True, duration=750, loop=0)
    else:
        fig.show()
# %% [markdown]
# ### Function for figure 3.

# %%
def plot_members_attendees(total_data, sns_cols, filename='members_attendees.png'):

    total_attendees = total_data.copy()
    low_adjs_members = lowess(total_data['Members'], np.arange(17))[:, 1]
    # Reshaping for seaborn
    total_attendees = total_attendees.melt(value_vars=['Members', 'Attendees'],
                                           id_vars=['Conference', 'Year'],
                                           value_name='Number', var_name='Group')

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    h3 = ax.plot(np.arange(17), low_adjs_members, '-*',
                 color=sns_cols[2], linewidth=4, markersize=10)
    h1 = ax.get_legend_handles_labels()

    x_labels = [ii.split(' ', 1)[1] for ii in total_data.Conference.values]

    sns.barplot(data=total_attendees, x='Conference', y='Number', hue='Group', ax=ax)

    h2 = ax.get_legend_handles_labels()[0]

    ax.legend(handles= h3 + h2, labels=['Lowess Trend', 'Members', 'Registrations'], loc='upper left')

    ax.set_xticklabels(x_labels, rotation=90, fontdict={'size': 20})

    ax.set_yticklabels(ax.get_yticklabels(), fontdict={'size': 20})

    ax.set(ylabel='', xlabel='')
    ax.set_title('Membership and Annual Meeting Registrations (2006 - 2022)', fontdict={'size': 32})

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')


# %% [markdown]
# ### Data preparation and setting a few default information by hand.

# %%
# Seaborn colors
sns_cols = sns.color_palette(n_colors=3)

# Handcrafting continents
conference_continent = ['Europe', 'North America', 'Oceania', 'North America', 'Europe',
                        'North America', 'Asia', 'North America', 'Europe',
                        'North America', 'Europe', 'North America',
                        'Asia', 'Europe', 'Virtual', 'Virtual', 'Europe']

# Handcrafting coordinates
conference_coordinates = {'2006 Florence': [43.769562, 11.255814, 'Florence'],
                          '2007 Chicago': [41.878113, -87.629799, 'Chicago'],
                          '2008 Melbourne': [-37.813629, 144.963058, 'Melbourne'],
                          '2009 San Francisco': [37.780079, -122.420174, 'San Francisco'],
                          '2010 Barcelona': [41.387920, 2.169920, 'Barcelona'],
                          '2011 Quebec City': [46.829853, -71.254028, 'Quebec City'],
                          '2012 Beijing': [39.906217,116.3912757, 'Beijing'],
                          '2013 Seattle': [47.6038321,-122.330062, 'Seattle'],
                          '2014 Hamburg': [53.550341,10.000654, 'Hamburg'],
                          '2015 Honolulu': [21.304547,-157.855676, 'Honolulu'],
                          '2016 Geneva': [46.2017559,6.1466014, 'Geneva'],
                          '2017 Vancouver': [49.2608724,-123.113952, 'Vancouver'],
                          '2018 Singapore': [1.357107,103.8194992, 'Singapore'],
                          '2019 Rome': [41.8933203,12.4829321, 'Rome'],
                          '2020 Virtual': [0, 0, 'Virtual'],
                          '2021 Virtual': [0, 0, 'Virtual'],
                          '2022 Glasgow': [55.8606182,-4.2497933, 'Glasgow']}
# %%
continent_data = pd.read_csv('continent_data.tsv', sep='\t')
total_data = pd.read_csv('total_data.tsv', sep='\t')
map_data = warp_continent_data_for_map(continent_data)
# %% [markdown]
# ## Plots

# %%
plot_members_by_continent(continent_data, conference_continent, sns_cols, None)
# %% [markdown]
# **Fig. 1**: OHBM membership data per year per continent. Bars in orange indicate if the Annual Meeting took place on the same continent, the virtual conferences are shaded in gray. Note different y-axis ranges for each plot. Numbers for North America include Central America and the Caribbean.

# %%
plot_animated_map(map_data, conference_coordinates, save_to_gif=False)
# %% [markdown]
# **Fig. 2**: Map of OHBM members by country of origin for each year (2006–2022). The location of the Annual Meeting is highlighted for each year. Note that light yellow includes 0; gray indicates countries for which no data is available (i.e., no OHBM members at any time).

# %%
plot_members_attendees(total_data, sns_cols, None)

# %% [markdown]
# **Fig. 3**: Total membership over the years, conference attendees for those years, and a LOWESS estimate of the general trend in membership numbers. 
