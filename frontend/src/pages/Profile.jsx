import {useState, useEffect} from "react";
import {useAuth} from "../features/Auth/AuthProvider";
import {secureFetch} from "../api/api.js";

const ProfilePage = () => {
    const { user, loadUserFromToken } = useAuth();
    const [firstName, setFirstName] = useState(user?.first_name || '');
    const [lastName, setLastName] = useState(user?.last_name || '');
    const [organization, setOrganization] = useState(user?.organization_name || '');
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);

    useEffect(() => {
        if (user) {
             setFirstName(user.first_name || '');
             setLastName(user.last_name || '');
             setOrganization(user.organization_name || '');
        }
    }, [user]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setMessage('');
        setError('');
        setIsSubmitting(true);

        try {
            const response = await secureFetch('/auth/users/me/', {
                method: 'PUT',
                body: JSON.stringify({
                    first_name: firstName,
                    last_name: lastName,
                    organization_name: organization
                }),
            });

            await response.json();
            await loadUserFromToken();
            setMessage('Профіль успішно оновлено!');
        } catch (e) {
            setError('Помилка оновлення профілю. Спробуйте пізніше.');
        }
        setIsSubmitting(false);
    };

    // Базові класи для інпутів у темному стилі
    const inputClass = "mt-1 block w-full px-3 py-2 border border-gray-600 rounded-lg bg-gray-700 text-white placeholder-gray-400 focus:ring-indigo-500 focus:border-indigo-500";


    return (
        // Оновлено для темної теми
        <div className="max-w-xl mx-auto p-6 my-10 bg-gray-800 rounded-xl shadow-2xl shadow-gray-700/30">
            <h2 className="text-3xl font-bold text-white mb-6 border-b border-gray-700 pb-3">Ваш Профіль</h2>

            {/* Оновлено для темної теми */}
            {message && <div className="p-3 mb-4 bg-green-900 text-green-400 rounded-lg">{message}</div>}
            {error && <div className="p-3 mb-4 bg-red-900 text-red-400 rounded-lg">{error}</div>}

            <form onSubmit={handleSubmit} className="space-y-4">
                {/* Оновлено для темної теми */}
                <div className="p-4 bg-gray-700 rounded-lg">
                    <label className="block text-sm font-medium text-gray-300">Електронна пошта (незмінна)</label>
                    <p className="mt-1 text-lg font-semibold text-white">{user?.email}</p>
                </div>
                 <div>
                    <label className="block text-sm font-medium text-gray-300">Ім'я</label>
                    <input type="text" value={firstName} onChange={(e) => setFirstName(e.target.value)}
                        className={inputClass}
                    />
                </div>
                 <div>
                    <label className="block text-sm font-medium text-gray-300">Прізвище</label>
                    <input type="text" value={lastName} onChange={(e) => setLastName(e.target.value)}
                        className={inputClass}
                    />
                </div>
                <div>
                    <label className="block text-sm font-medium text-gray-300">Організація/Компанія</label>
                    <input type="text" value={organization} onChange={(e) => setOrganization(e.target.value)}
                        className={inputClass}
                    />
                </div>
                <div>
                    <label className="block text-sm font-medium text-gray-300">Роль</label>
                    <p className="mt-1 font-medium text-indigo-400">{user?.role || 'FARMER'}</p>
                </div>

                <button
                    type="submit"
                    disabled={isSubmitting}
                    className="w-full py-2 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 transition"
                >
                    {isSubmitting ? 'Оновлення...' : 'Зберегти зміни'}
                </button>
            </form>
        </div>
    );
};

export default ProfilePage;