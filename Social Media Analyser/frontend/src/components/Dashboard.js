import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { getClientStats } from '../services/api';

export default function Dashboard() {
  const [stats, setStats] = useState([]);

  useEffect(() => {
    getClientStats('client_1').then(data => setStats(data));
  }, []);

  const labels = stats.map(row => row.date);
  const followers = stats.map(row => row.followers);

  const data = {
    labels,
    datasets: [
      {
        label: 'Followers Growth',
        data: followers,
        borderColor: 'blue',
        fill: false,
      },
    ],
  };

  return (
    <div>
      <h2>Social Media Analytics Dashboard</h2>
      <Line data={data} />
    </div>
  );
}
