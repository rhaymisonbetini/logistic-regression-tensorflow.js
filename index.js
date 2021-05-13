const csvUrl = 'https://gist.githubusercontent.com/juandes/ba58ef99df9bd719f87f807e24f7ea1c/raw/59f57af034c52bd838c513563a3e547b3650e7ba/lr-dataset.csv';
let dataset;

function loadData() {
    dataset = tf.data.csv(
        csvUrl, {
        columnConfigs: {
            label: {
                isLabel: true,
            },
        },
    },
    );
}

async function visualizeDataset() {
    const dataSurface = { name: 'Scatterplot', tab: 'Charts' };
    const classZero = [];
    const classOne = [];

    await dataset.forEachAsync((e) => {
        const features = { x: e.xs.feature_1, y: e.xs.feature_2 };
        if (e.ys.label === 0) {
            classZero.push(features);
        } else {
            classOne.push(features);
        }
    });

    const series = ['Class 0', 'Class 1'];
    const dataToDraw = { values: [classZero, classOne], series };

    tfvis.render.scatterplot(dataSurface, dataToDraw, {
        xLabel: 'feature_1',
        yLabel: 'feature_2',
        zoomToFit: true,
    });
}

function createVisualizeButton() {
    const btn = document.createElement('BUTTON');
    btn.innerText = 'Visualize!';
    btn.addEventListener('click', () => {
        visualizeDataset();
    });
    document.querySelector('#visualize').appendChild(btn);
}

function init() {
    createVisualizeButton();
    loadData();
}

init();