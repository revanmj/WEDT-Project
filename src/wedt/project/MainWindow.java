/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wedt.project;

import java.io.File;
import java.util.List;
import javax.swing.DefaultListModel;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import twitter4j.*;
import twitter4j.conf.ConfigurationBuilder;

/**
 *
 * @author revanmj
 */
public class MainWindow extends javax.swing.JFrame {

    private BayesClassifier bayesC;
    private SvmClassifier svmC;
    private Common cmn;
    private DefaultListModel listModel;
    private boolean trained = false;
    
    /**
     * Creates new form MainWindow
     */
    public MainWindow() {
        initComponents();
        bayesC = new BayesClassifier();
        svmC = new SvmClassifier();
        cmn = new Common();
        
        try {
            File bayesData = new File("Bayes.model");
            File svmData = new File("SVM.model");
            if (bayesData.exists() && !svmData.exists())
                trained = true;
        } catch (Exception e) {
            System.out.println("init failed");
        }
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        fileChooser = new javax.swing.JFileChooser();
        jTextField1 = new javax.swing.JTextField();
        searchButton = new javax.swing.JButton();
        checkBoxPopular = new javax.swing.JCheckBox();
        checkBoxLatest = new javax.swing.JCheckBox();
        jScrollPane1 = new javax.swing.JScrollPane();
        tweetsList = new javax.swing.JList();
        checkTweetButton = new javax.swing.JButton();
        testFromCsvButton = new javax.swing.JButton();
        jLabel1 = new javax.swing.JLabel();
        bayesLabel = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();
        svmLabel = new javax.swing.JLabel();
        statusLabel = new javax.swing.JLabel();
        learnButton = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        searchButton.setText("Szukaj");
        searchButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                searchButtonActionPerformed(evt);
            }
        });

        checkBoxPopular.setText("Popularne");

        checkBoxLatest.setText("Najnowsze");
        checkBoxLatest.setName("checkBoxLatest"); // NOI18N

        jScrollPane1.setViewportView(tweetsList);

        checkTweetButton.setText("Ocena");
        checkTweetButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                checkTweetButtonActionPerformed(evt);
            }
        });

        testFromCsvButton.setText("Test danymi z pliku CSV");
        testFromCsvButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                testFromCsvButtonActionPerformed(evt);
            }
        });

        jLabel1.setText("Bayes:");

        jLabel2.setText("SVM:");

        learnButton.setText("Uczenie");
        learnButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                learnButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 560, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(jLabel1)
                            .addComponent(bayesLabel)
                            .addComponent(jLabel2)
                            .addComponent(svmLabel)
                            .addComponent(learnButton, javax.swing.GroupLayout.DEFAULT_SIZE, 101, Short.MAX_VALUE)
                            .addComponent(checkTweetButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addGap(0, 11, Short.MAX_VALUE))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addGroup(layout.createSequentialGroup()
                                        .addComponent(checkBoxLatest)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                        .addComponent(checkBoxPopular)
                                        .addGap(0, 0, Short.MAX_VALUE))
                                    .addComponent(jTextField1))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(searchButton))
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(testFromCsvButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(statusLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 317, javax.swing.GroupLayout.PREFERRED_SIZE)))
                        .addContainerGap())))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jTextField1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(searchButton))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(checkBoxPopular)
                    .addComponent(checkBoxLatest))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 273, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jLabel1)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(bayesLabel)
                        .addGap(18, 18, 18)
                        .addComponent(jLabel2)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(svmLabel)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(learnButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(checkTweetButton)
                        .addGap(0, 135, Short.MAX_VALUE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(testFromCsvButton)
                    .addComponent(statusLabel))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void searchButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_searchButtonActionPerformed
        ConfigurationBuilder cb = new ConfigurationBuilder();
        cb.setDebugEnabled(true)
            .setOAuthConsumerKey("PG0vtiQ73sbKKCfp9JfqyQ")
            .setOAuthConsumerSecret("ITCkTQiqCh3aVZexXentwnwCJooVpUOcpkIENPKowI")
            .setOAuthAccessToken("89783194-z0J1KLudg6MFMhhysKmL29zB5wBjxfxWUboAh6lAI")
            .setOAuthAccessTokenSecret("ytOdt7t8P1OrmAI2ZCRoX30ZC3eLcDSgPY8gOa6FCwQ");
        TwitterFactory tf = new TwitterFactory(cb.build());
        Twitter twitter = tf.getInstance();
//        try {
//            Map<String ,RateLimitStatus> rateLimitStatus = twitter.getRateLimitStatus();
//            for (String endpoint : rateLimitStatus.keySet()) {
//                RateLimitStatus status = rateLimitStatus.get(endpoint);
//                System.out.println("Endpoint: " + endpoint);
//                System.out.println(" Limit: " + status.getLimit());
//                System.out.println(" Remaining: " + status.getRemaining());
//                System.out.println(" ResetTimeInSeconds: " + status.getResetTimeInSeconds());
//                System.out.println(" SecondsUntilReset: " + status.getSecondsUntilReset());
//            }
//        } catch (TwitterException te) {
//            te.printStackTrace();
//            System.out.println("Failed to get rate limit status: " + te.getMessage());
//        }
        try {
            statusLabel.setText("Trwa wyszukiwanie...");
            Query query = new Query(jTextField1.getText());
            query.setCount(5);
            query.setLang("en");
            if (checkBoxLatest.isSelected() && checkBoxPopular.isSelected())
                query.setResultType(Query.ResultType.mixed);
            else if (checkBoxLatest.isSelected())
                query.setResultType(Query.ResultType.recent);
            else if (checkBoxPopular.isSelected())
                query.setResultType(Query.ResultType.popular);
            QueryResult result;

            result = twitter.search(query);
            List<Status> tweets = result.getTweets();
            listModel = new DefaultListModel();
            tweetsList.setModel(listModel);
            tweets.stream().forEach((tweet) -> {
                listModel.addElement(tweet.getText());
            });
                
        } catch (TwitterException te) {
            statusLabel.setText("Wyszukiwanie nie powiodlo sie");
            te.printStackTrace();
            //System.out.println("Failed to search tweets: " + te.getMessage());
            JOptionPane.showMessageDialog(null, te.getMessage(), "Blad pobierania wynikow wyszukiwania", JOptionPane.INFORMATION_MESSAGE);
        }
    }//GEN-LAST:event_searchButtonActionPerformed

    private void learnButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_learnButtonActionPerformed
        int returnVal = fileChooser.showOpenDialog(this);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            try {
                statusLabel.setText("Trwa uczenie (Bayes) ...");
                bayesC.train(file, cmn);
                //statusLabel.setText("Trwa uczenie (SVM) ...");
                //svmC.train(file, cmn);
                statusLabel.setText("Gotowe");
                trained = true;
            } catch (Exception ex) {
                statusLabel.setText("Blad");
                System.out.println("problem accessing file "+file.getAbsolutePath());
                System.out.println(ex.toString());
            }
        }
    }//GEN-LAST:event_learnButtonActionPerformed

    private void checkTweetButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_checkTweetButtonActionPerformed
        if (trained) {
            String bayesResult;
            String svmResult;
            statusLabel.setText("Trwa klasyfikowanie (Bayes) ...");
            bayesResult = bayesC.classifySingle(listModel.get(tweetsList.getSelectedIndex()).toString(), cmn);
            //statusLabel.setText("Trwa klasyfikowanie (SVM) ...");
            //svmResult = svmC.classifySingle(listModel.get(tweetsList.getSelectedIndex()).toString(), cmn);
            statusLabel.setText("Gotowe");
            bayesLabel.setText(bayesResult);
            //svmLabel.setText(svmResult);
        } else
            JOptionPane.showMessageDialog(null, "Najpierw uruchom proces uczenia!", "Blad!", JOptionPane.INFORMATION_MESSAGE);
    }//GEN-LAST:event_checkTweetButtonActionPerformed

    private void testFromCsvButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_testFromCsvButtonActionPerformed
        int returnVal = fileChooser.showOpenDialog(this);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            try {
                statusLabel.setText("Trwa klasyfikowanie (Bayes) ...");
                int bayesErr = bayesC.classifyFromCsv(file, cmn);
                //statusLabel.setText("Trwa klasyfikowanie (SVM) ...");
                //int svmErr = svmC.classifyFromCsv(file, cmn);
                statusLabel.setText("Gotowe");
                JOptionPane.showMessageDialog(null, "Bledy Bayes: " + bayesErr + "\nBledy SVM: " + 0, "Wynik testu", JOptionPane.INFORMATION_MESSAGE);
            } catch (Exception ex) {
                System.out.println("problem accessing file"+file.getAbsolutePath());
            }
            statusLabel.setText("Blad");
        }
    }//GEN-LAST:event_testFromCsvButtonActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(MainWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(MainWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(MainWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(MainWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(() -> {
            new MainWindow().setVisible(true);
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel bayesLabel;
    private javax.swing.JCheckBox checkBoxLatest;
    private javax.swing.JCheckBox checkBoxPopular;
    private javax.swing.JButton checkTweetButton;
    private javax.swing.JFileChooser fileChooser;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JTextField jTextField1;
    private javax.swing.JButton learnButton;
    private javax.swing.JButton searchButton;
    private javax.swing.JLabel statusLabel;
    private javax.swing.JLabel svmLabel;
    private javax.swing.JButton testFromCsvButton;
    private javax.swing.JList tweetsList;
    // End of variables declaration//GEN-END:variables
}
