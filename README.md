# Analiza wydźwięku tweetów

![My image](https://github.com/revanmj/wedt-project/raw/master/screenshot.png)

Aplikacja służy do oceny wydźwięku (sentiment) tweetów przy pomocy klasyfikatorów Bayesa i SMO oferowanych przez bibliotekę WEKA.

Oferowane funkcje:
- Ocena pojedynczych tweetów odnalezionych przy pomocy wbudowanej wyszukiwarki
- Masowa ocena tweetów z plików CSV (dokładne wyniki są wyświetlane na konsoli)
- Ustawianie własnych parametrów klasyfikatorów (Listy parametrów: [Bayes](http://weka.sourceforge.net/doc.dev/weka/classifiers/bayes/NaiveBayes.html), [SMO](http://weka.sourceforge.net/doc.dev/weka/classifiers/functions/SMO.html))

Format plików CSV ze zbiorami uczącymi oraz testowymi:

<pre><code>Tweet,Sentiment
"treść_tweeta1",positive
"treść_tweeta2",negative
"treść_tweeta3",neutral
...
</code></pre>

Aplikacja powstała w środowisku NetBeans 8.0.
