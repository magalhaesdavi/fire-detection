import seaborn as sns

d = {'Rudz': list(range(generation_count + 1)), 'Valor de fitness médio': avg_fitness_lst}
        df = pd.DataFrame(d)

        sns.set(font_scale=1.2, style="whitegrid")

        fitness_plot = sns.relplot(
            x='Geração',
            y='Valor de fitness médio',
            data=df,
            kind="line",
            lw=3,
            palette=algo_colors,
            markers=True,
            dashes=False
        )
        fitness_plot.set_axis_labels("Geração", "Valor de fitness médio")
        fitness_plot.fig.suptitle('Gráfico da função fitness')
        fitness_plot.fig.title_fontsize = 18
        fitness_plot.fig.set_size_inches((12, 8))
        fitness_plot.savefig('./results/' + level + '/fitness_plot.png')