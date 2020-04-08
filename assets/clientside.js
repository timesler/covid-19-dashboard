window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        update_range: function(value, date, fig_in) {
            if (fig_in === undefined) {
                return fig_in
            }

            // Get new upper bound for x
            var max_x = new Date();
            max_x.setDate(max_x.getDate() + value);

            // Get max y for which x is <= max_x
            var x = fig_in['data'][1]['x'];
            var y = fig_in['data'][1]['y'];
            var max_y = -1000000;
            for (var i = 0; i < x.length; i++) {
                var x_date = new Date(x[i]);
                if (x_date < max_x) {
                    if (y[i] > max_y) max_y = y[i];
                }
            }

            var max_y = Math.max(max_y, Math.max(...fig_in['data'][0]['y'])) * 1.05;

            var max_x = max_x
                .toLocaleString('en-us', {year: 'numeric', month: '2-digit', day: '2-digit'})
                .replace(/(\d+)\/(\d+)\/(\d+)/, '$3-$1-$2T00:00:00');

            var x_range = [
                date, max_x
            ];
            var y_range = [
                -0.02 * max_y, 1.02 * max_y
            ];

            const fig = Object.assign({}, fig_in, {
                'layout': {
                    ...fig_in.layout,
                    'yaxis': {
                        ...fig_in.layout.yaxis, range: y_range
                    },
                    'xaxis': {
                        ...fig_in.layout.xaxis, range: x_range
                    }
                 }
            });

            return fig
        }
    }
});
